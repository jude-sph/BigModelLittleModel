package com.jeeves

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.graphics.Rect
import android.util.Log
import android.view.Display
import android.view.WindowInsets
import android.view.WindowManager
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import android.view.accessibility.AccessibilityWindowInfo
import android.content.Intent
import com.jeeves.model.ElementNode
import com.jeeves.model.PhoneState
import android.os.Handler
import android.os.Looper
import java.util.concurrent.atomic.AtomicBoolean
import android.graphics.Bitmap
import android.util.Base64
import java.io.ByteArrayOutputStream
import java.util.concurrent.CompletableFuture
import android.accessibilityservice.GestureDescription
import android.graphics.Path

class JeevesAccessibilityService : AccessibilityService(), ConfigManager.ConfigChangeListener {

    companion object {
        private const val TAG = "JeevesA11yService"
        private var instance: JeevesAccessibilityService? = null
        private const val MIN_ELEMENT_SIZE = 5

        // Periodic update constants
        private const val REFRESH_INTERVAL_MS = 250L // Update every 250ms
        private const val MIN_FRAME_TIME_MS = 16L // Minimum time between frames (roughly 60 FPS)

        fun getInstance(): JeevesAccessibilityService? = instance
    }

    private lateinit var overlayManager: OverlayManager
    private val screenBounds = Rect()
    private lateinit var configManager: ConfigManager
    private val mainHandler = Handler(Looper.getMainLooper())
    private var socketServer: SocketServer? = null

    // Periodic update state
    private var isInitialized = false
    private val isProcessing = AtomicBoolean(false)
    private var lastUpdateTime = 0L
    private var currentPackageName: String = ""
    private val visibleElements = mutableListOf<ElementNode>()
    private val indexedElements = mutableMapOf<Int, ElementNode>() // Map of index -> ElementNode for consistent indexing

    override fun onCreate() {
        super.onCreate()
        overlayManager = OverlayManager(this)
        val windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        val windowMetrics = windowManager.currentWindowMetrics
        val bounds = windowMetrics.bounds
        screenBounds.set(0, 0, bounds.width(), bounds.height())

        // Calculate and set coordinate system offset to fix mismatch between
        // getBoundsInScreen() (uses full display coordinates) and overlay drawing (uses app content coordinates)
        val coordinateOffset = calculateCoordinateOffset(windowManager)
        overlayManager.setPositionOffsetY(-coordinateOffset) // Negative to shift up

        // Initialize ConfigManager
        configManager = ConfigManager.getInstance(this)
        configManager.addListener(this)

        // Initialize SocketServer
        socketServer = SocketServer(this)

        isInitialized = true
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        overlayManager.showOverlay()
        instance = this

        // Configure accessibility service
        serviceInfo = AccessibilityServiceInfo().apply {
            // Listen to all events
            eventTypes = AccessibilityEvent.TYPES_ALL_MASK

            // Monitor all packages
            packageNames = null

            // Set feedback type
            feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC

            // Set flags for better access
            flags = AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS or
                    AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS or
                    AccessibilityServiceInfo.FLAG_REQUEST_TOUCH_EXPLORATION_MODE

            // Enable screenshot capability (API 34+)
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                flags = flags or AccessibilityServiceInfo.FLAG_REQUEST_2_FINGER_PASSTHROUGH
            }
        }

        // Apply loaded configuration
        applyConfiguration()

        // Start periodic updates
        startPeriodicUpdates()

        // Start socket server if enabled
        startSocketServerIfEnabled()

        Log.d(TAG, "Accessibility service connected and configured")
        sendAccessibilityServiceStatusBroadcast(true)
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        val eventPackage = event?.packageName?.toString() ?: ""

        // Detect package changes
        if (eventPackage.isNotEmpty() && eventPackage != currentPackageName && currentPackageName.isNotEmpty()) {
            Log.d(TAG, "Package changed from $currentPackageName to $eventPackage - resetting state")
            resetOverlayState()
        }

        if (eventPackage.isNotEmpty()) {
            currentPackageName = eventPackage
        }

        // Trigger update on relevant events - be more aggressive about updates
        when (event?.eventType) {
            AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED,
            AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED,
            AccessibilityEvent.TYPE_VIEW_SCROLLED,
            AccessibilityEvent.TYPE_VIEW_CLICKED,
            AccessibilityEvent.TYPE_VIEW_FOCUSED -> {
                Log.d(TAG, "UI event detected: ${event.eventType} - refreshing elements")
                refreshVisibleElements()
            }
        }
    }

    // Periodic update runnable
    private val updateRunnable = object : Runnable {
        override fun run() {
            if (isInitialized && configManager.overlayVisible) {
                val currentTime = System.currentTimeMillis()
                val timeSinceLastUpdate = currentTime - lastUpdateTime

                if (timeSinceLastUpdate >= MIN_FRAME_TIME_MS) {
                    refreshVisibleElements()
                    lastUpdateTime = currentTime
                }
            }
            mainHandler.postDelayed(this, REFRESH_INTERVAL_MS)
        }
    }

    private fun startPeriodicUpdates() {
        lastUpdateTime = System.currentTimeMillis()
        mainHandler.postDelayed(updateRunnable, REFRESH_INTERVAL_MS)
        Log.d(TAG, "Started periodic updates")
    }

    private fun stopPeriodicUpdates() {
        mainHandler.removeCallbacks(updateRunnable)
        Log.d(TAG, "Stopped periodic updates")
    }

    private fun getValidRootNode(): AccessibilityNodeInfo? {
        val rootNode = rootInActiveWindow
        val rootPackage = rootNode?.packageName?.toString()
        val currentWindows = windows

        val rootPackageHasWindow = currentWindows?.any { window ->
            window.root?.packageName?.toString() == rootPackage
        } ?: false

        // if rootInActiveWindow is valid and has a matching window, use it
        if (rootNode != null && rootPackageHasWindow) {
            return rootNode
        }

        // rootInActiveWindow is null or stale - get root from windows list instead
        Log.d(TAG, "Using fallback: rootInActiveWindow ($rootPackage) not in windows list")

        // find the best window to use - prefer TYPE_APPLICATION, fallback to others
        val fallbackRoot = currentWindows
            ?.sortedByDescending { window ->
                when (window.type) {
                    AccessibilityWindowInfo.TYPE_APPLICATION -> 2
                    AccessibilityWindowInfo.TYPE_SYSTEM -> 1
                    else -> 0
                }
            }
            ?.firstNotNullOfOrNull { it.root }

        if (fallbackRoot != null) {
            Log.d(TAG, "Fallback root: ${fallbackRoot.packageName}")
        }

        return fallbackRoot
    }
    private fun refreshVisibleElements() {
        if (!isProcessing.compareAndSet(false, true)) {
            return
        }

        try {
            var rootNode = getValidRootNode()
            if (currentPackageName.isEmpty()) {
                overlayManager.clearElements()
                overlayManager.refreshOverlay()
                indexedElements.clear()
                return
            }

            // Clear previous elements
            clearElementList()
            indexedElements.clear()

            // Get fresh elements
            val elements = getVisibleElementsInternal()

            // Collect all elements in a flat list first
            val allElements = mutableListOf<ElementNode>()
            elements.forEach { root ->
                collectAllElements(root, allElements)
            }

            // Assign sequential indices to ALL elements
            allElements.forEachIndexed { index, element ->
                element.overlayIndex = index
                indexedElements[index] = element
            }

            // Update overlay if visible
            if (configManager.overlayVisible && allElements.isNotEmpty()) {
                overlayManager.clearElements()

                // Add only elements that should be shown
                allElements.forEach { element ->
                    if (shouldShowInOverlay(element)) {
                        overlayManager.addElement(
                            text = element.text,
                            rect = element.rect,
                            type = element.className,
                            index = element.overlayIndex
                        )
                    }
                }

                overlayManager.refreshOverlay()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error refreshing visible elements: ${e.message}", e)
        } finally {
            isProcessing.set(false)
        }
    }

    private fun resetOverlayState() {
        try {
            overlayManager.clearElements()
            overlayManager.refreshOverlay()
            clearElementList()
            indexedElements.clear() // Also clear the index map
            Log.d(TAG, "Reset overlay state for package change")
        } catch (e: Exception) {
            Log.e(TAG, "Error resetting overlay state: ${e.message}", e)
        }
    }

    private fun clearElementList() {
        for (element in visibleElements) {
            try {
                element.nodeInfo.recycle()
            } catch (e: Exception) {
                Log.e(TAG, "Error recycling node: ${e.message}")
            }
        }
        visibleElements.clear()
    }

    private fun applyConfiguration() {
        mainHandler.post {
            try {
                val config = configManager.getCurrentConfiguration()
                if (config.overlayVisible) {
                    overlayManager.showOverlay()
                } else {
                    overlayManager.hideOverlay()
                }
                // DON'T override the dynamic offset on initial setup - only apply user adjustments
                if (config.overlayOffset != 0) {
                    overlayManager.setPositionOffsetY(config.overlayOffset)
                    Log.d(TAG, "Applied user offset: ${config.overlayOffset}")
                } else {
                    Log.d(TAG, "Keeping dynamic offset, no user adjustment")
                }
                Log.d(TAG, "Applied configuration: overlayVisible=${config.overlayVisible}")
            } catch (e: Exception) {
                Log.e(TAG, "Error applying configuration: ${e.message}", e)
            }
        }
    }

    // Public methods for MainActivity to call directly
    fun setOverlayVisible(visible: Boolean): Boolean {
        return try {
            configManager.overlayVisible = visible

            mainHandler.post {
                if (visible) {
                    overlayManager.showOverlay()
                    // Trigger immediate refresh when showing overlay
                    refreshVisibleElements()
                } else {
                    overlayManager.hideOverlay()
                }
            }

            Log.d(TAG, "Overlay visibility set to: $visible")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error setting overlay visibility: ${e.message}", e)
            false
        }
    }

    fun isOverlayVisible(): Boolean = configManager.overlayVisible

    fun setOverlayOffset(offset: Int): Boolean {
        return try {
            configManager.overlayOffset = offset

            mainHandler.post {
                overlayManager.setPositionOffsetY(offset)
            }

            Log.d(TAG, "Overlay offset set to: $offset")
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error setting overlay offset: ${e.message}", e)
            false
        }
    }

    fun getOverlayOffset(): Int = configManager.overlayOffset

    fun getVisibleElements(): MutableList<ElementNode> {
        return getVisibleElementsInternal()
    }

    private fun getVisibleElementsInternal(): MutableList<ElementNode> {
        val elements = mutableListOf<ElementNode>()
        val indexCounter = IndexCounter(0) // Start indexing from 0 to match Python

        val rootNode = getValidRootNode() ?: return elements // used to be rootInActiveWindow
        val rootElement = findAllVisibleElements(rootNode, 0, null, indexCounter)
        rootElement?.let {
            collectRootElements(it, elements)
        }

        // Store the elements for cleanup later
        synchronized(visibleElements) {
            clearElementList()
            visibleElements.addAll(elements)
        }

        return elements
    }

    private fun collectRootElements(element: ElementNode, rootElements: MutableList<ElementNode>) {
        rootElements.add(element)
    }

    private fun findAllVisibleElements(
        node: AccessibilityNodeInfo,
        windowLayer: Int,
        parent: ElementNode?,
        indexCounter: IndexCounter
    ): ElementNode? {
        try {

            val rect = Rect()
            node.getBoundsInScreen(rect)

            val isInScreen = Rect.intersects(rect, screenBounds)
            val hasSize = rect.width() > MIN_ELEMENT_SIZE && rect.height() > MIN_ELEMENT_SIZE

            var currentElement: ElementNode? = null

            if (isInScreen && hasSize) {
                val text = node.text?.toString() ?: ""
                val contentDesc = node.contentDescription?.toString() ?: ""
                val className = node.className?.toString() ?: ""
                val viewId = node.viewIdResourceName ?: ""

                val displayText = when {
                    text.isNotEmpty() -> text
                    contentDesc.isNotEmpty() -> contentDesc
                    viewId.isNotEmpty() -> viewId.substringAfterLast('/')
                    else -> className.substringAfterLast('.')
                }

                val elementType = if (node.isClickable) {
                    "Clickable"
                } else if (node.isCheckable) {
                    "Checkable"
                } else if (node.isEditable) {
                    "Input"
                } else if (text.isNotEmpty()) {
                    "Text"
                } else if (node.isScrollable) {
                    "Container"
                } else {
                    "View"
                }

                val id = ElementNode.createId(rect, className.substringAfterLast('.'), displayText)

                currentElement = ElementNode(
                    AccessibilityNodeInfo(node),
                    Rect(rect),
                    displayText,
                    className.substringAfterLast('.'),
                    windowLayer,
                    System.currentTimeMillis(),
                    id
                )

                // Assign unique index
                currentElement.overlayIndex = indexCounter.getNext()

                // Set parent-child relationship
                parent?.addChild(currentElement)
            }

            // Recursively process children
            for (i in 0 until node.childCount) {
                val childNode = node.getChild(i) ?: continue
                val childElement = findAllVisibleElements(childNode, windowLayer, currentElement, indexCounter)
                // Children are already added to currentElement via parent?.addChild() call above
            }

            return currentElement

        } catch (e: Exception) {
            Log.e(TAG, "Error in findAllVisibleElements: ${e.message}", e)
            return null
        }
    }

     fun getPhoneState(): PhoneState {
        val focusedNode = findFocus(AccessibilityNodeInfo.FOCUS_INPUT) ?: findFocus(AccessibilityNodeInfo.FOCUS_ACCESSIBILITY)
        val keyboardVisible = detectKeyboardVisibility()
        val currentPackage = getValidRootNode()?.packageName?.toString() // used to be rootInActiveWindow
        val appName = getAppName(currentPackage)

        return PhoneState(focusedNode, keyboardVisible, currentPackage, appName)
    }

    private fun detectKeyboardVisibility(): Boolean {
        try {
            val windows = windows
            if (windows != null) {
                val hasInputMethodWindow = windows.any { window -> window.type == AccessibilityWindowInfo.TYPE_INPUT_METHOD }
                windows.forEach { it.recycle() }
                return hasInputMethodWindow
            } else { return false }
        } catch (e: Exception) { return false}
    }

    private fun getAppName(packageName: String?): String? {
        return try {
            if (packageName == null) return null

            val packageManager = packageManager
            val applicationInfo = packageManager.getApplicationInfo(packageName, 0)
            packageManager.getApplicationLabel(applicationInfo).toString()
        } catch (e: Exception) {
            Log.e(TAG, "Error getting app name for package $packageName: ${e.message}")
            null
        }
    }

    // Helper class to maintain global index counter
    private class IndexCounter(private var current: Int = 0) {
        fun getNext(): Int = current++
    }

    // Collect all elements in a flat list (depth-first traversal)
    private fun collectAllElements(element: ElementNode, allElements: MutableList<ElementNode>) {
        allElements.add(element)
        element.children.forEach { child ->
            collectAllElements(child, allElements)
        }
    }

    // Determine if an element should be shown in the overlay
    private fun shouldShowInOverlay(element: ElementNode): Boolean {
        // Show clickable, checkable, editable elements and non-empty text
        return element.nodeInfo.isClickable ||
               element.nodeInfo.isCheckable ||
               element.nodeInfo.isEditable ||
               (element.text.isNotEmpty() && element.rect.width() > MIN_ELEMENT_SIZE && element.rect.height() > MIN_ELEMENT_SIZE)
    }

    // Get element by its index (for tap operations)
    fun getElementByIndex(index: Int): ElementNode? {
        return indexedElements[index]
    }

    // Get all indexed elements (for ContentProvider)
    fun getIndexedElements(): Map<Int, ElementNode> {
        return indexedElements.toMap()
    }

    // Force refresh elements (can be called from ContentProvider/SocketServer)
    fun forceRefreshElements() {
        Log.d(TAG, "Force refresh requested")
        refreshVisibleElements()
    }

    // Perform tap on element by index
    fun performTapByIndex(index: Int): String {
        val element = indexedElements[index]
            ?: return "error: No element with index $index. Available indices: ${indexedElements.keys.sorted().take(20).joinToString(", ")}"

        // Calculate center point
        val centerX = (element.rect.left + element.rect.right) / 2
        val centerY = (element.rect.top + element.rect.bottom) / 2

        // Create gesture for tap
        val gestureBuilder = GestureDescription.Builder()
        val path = Path().apply {
            moveTo(centerX.toFloat(), centerY.toFloat())
        }
        
        val gesture = gestureBuilder.addStroke(
            GestureDescription.StrokeDescription(path, 0, 50) // 50ms tap duration
        ).build()

        // Use CountDownLatch to wait for gesture completion
        val latch = java.util.concurrent.CountDownLatch(1)
        var gestureResult = "timeout"
        
        dispatchGesture(gesture, object : GestureResultCallback() {
            override fun onCompleted(gestureDescription: GestureDescription?) {
                gestureResult = "success"
                latch.countDown()
            }
            override fun onCancelled(gestureDescription: GestureDescription?) {
                gestureResult = "cancelled"
                latch.countDown()
            }
        }, null)

        // Wait for gesture to complete (max 500ms)
        try {
            latch.await(500, java.util.concurrent.TimeUnit.MILLISECONDS)
        } catch (e: InterruptedException) {
            Log.e(TAG, "Gesture wait interrupted", e)
        }

        // Force refresh elements after tap to ensure UI state is updated
        mainHandler.postDelayed({
            Log.d(TAG, "Waiting for UI to stabilize after tap")
            forceRefreshElements()
        }, 450)

        return if (gestureResult == "success") {
            "success: Tapped element $index at ($centerX, $centerY) - ${element.text}"
        } else {
            "error: Tap gesture failed for element $index at ($centerX, $centerY) - result: $gestureResult"
        }
    }

    // Socket server management methods
    private fun startSocketServerIfEnabled() {
        if (configManager.socketServerEnabled) {
            startSocketServer()
        }
    }

    private fun startSocketServer() {
        socketServer?.let { server ->
            if (!server.isRunning()) {
                val port = configManager.socketServerPort
                val success = server.start(port)
                if (success) {
                    Log.i(TAG, "Socket server started on port $port")
                } else {
                    Log.e(TAG, "Failed to start socket server on port $port")
                }
            }
        }
    }

    private fun stopSocketServer() {
        socketServer?.let { server ->
            if (server.isRunning()) {
                server.stop()
                Log.i(TAG, "Socket server stopped")
            }
        }
    }

    fun getSocketServerStatus(): String {
        return socketServer?.let { server ->
            if (server.isRunning()) {
                "Running on port ${server.getPort()}"
            } else {
                "Stopped"
            }
        } ?: "Not initialized"
    }

    fun getAdbForwardCommand(): String {
        val port = configManager.socketServerPort
        return "adb forward tcp:$port tcp:$port"
    }

    // ConfigManager.ConfigChangeListener implementation
    override fun onOverlayVisibilityChanged(visible: Boolean) {
        // Already handled in setOverlayVisible method
    }

    override fun onOverlayOffsetChanged(offset: Int) {
        // Already handled in setOverlayOffset method
    }

    override fun onSocketServerEnabledChanged(enabled: Boolean) {
        if (enabled) {
            startSocketServer()
        } else {
            stopSocketServer()
        }
    }

    override fun onSocketServerPortChanged(port: Int) {
        // Restart server with new port if it's running
        socketServer?.let { server ->
            if (server.isRunning()) {
                server.stop()
                val success = server.start(port)
                if (success) {
                    Log.i(TAG, "Socket server restarted on new port $port")
                } else {
                    Log.e(TAG, "Failed to restart socket server on new port $port")
                }
            }
        }
    }

    // Screenshot functionality
    fun takeScreenshotBase64(hideOverlay: Boolean = true): CompletableFuture<String> {
        val future = CompletableFuture<String>()

        // Temporarily hide overlay if requested
        val wasOverlayDrawingEnabled = if (hideOverlay) {
            val enabled = overlayManager.isDrawingEnabled()
            overlayManager.setDrawingEnabled(false)
            enabled
        } else {
            true
        }

        try {
            if (hideOverlay) {
                // Small delay to ensure overlay is hidden before screenshot
                mainHandler.postDelayed({
                    performScreenshotCapture(future, wasOverlayDrawingEnabled, hideOverlay)
                }, 100)
            } else {
                performScreenshotCapture(future, wasOverlayDrawingEnabled, hideOverlay)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error taking screenshot", e)
            future.complete("error: Failed to take screenshot: ${e.message}")

            // Restore overlay drawing state in case of exception
            if (hideOverlay) {
                overlayManager.setDrawingEnabled(wasOverlayDrawingEnabled)
            }
        }

        return future
    }

    private fun performScreenshotCapture(future: CompletableFuture<String>, wasOverlayDrawingEnabled: Boolean, hideOverlay: Boolean) {
        try {
            takeScreenshot(
                Display.DEFAULT_DISPLAY,
                mainHandler.looper.thread.contextClassLoader?.let {
                    java.util.concurrent.Executors.newSingleThreadExecutor()
                } ?: java.util.concurrent.Executors.newSingleThreadExecutor(),
                object : TakeScreenshotCallback {
                    override fun onSuccess(screenshotResult: ScreenshotResult) {
                        try {
                            val bitmap = Bitmap.wrapHardwareBuffer(
                                screenshotResult.hardwareBuffer,
                                screenshotResult.colorSpace
                            )

                            if (bitmap == null) {
                                Log.e(TAG, "Failed to create bitmap from hardware buffer")
                                screenshotResult.hardwareBuffer.close()
                                future.complete("error: Failed to create bitmap from screenshot data")
                                return
                            }

                            val byteArrayOutputStream = ByteArrayOutputStream()
                            val compressionSuccess = bitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream)

                            if (!compressionSuccess) {
                                Log.e(TAG, "Failed to compress bitmap to PNG")
                                bitmap.recycle()
                                screenshotResult.hardwareBuffer.close()
                                byteArrayOutputStream.close()
                                future.complete("error: Failed to compress screenshot to PNG format")
                                return
                            }

                            val byteArray = byteArrayOutputStream.toByteArray()
                            val base64String = Base64.encodeToString(byteArray, Base64.NO_WRAP)

                            bitmap.recycle()
                            screenshotResult.hardwareBuffer.close()
                            byteArrayOutputStream.close()

                            future.complete(base64String)
                            Log.d(TAG, "Screenshot captured successfully, size: ${byteArray.size} bytes")
                        } catch (e: Exception) {
                            Log.e(TAG, "Error processing screenshot", e)
                            try {
                                screenshotResult.hardwareBuffer.close()
                            } catch (closeException: Exception) {
                                Log.e(TAG, "Error closing hardware buffer", closeException)
                            }
                            future.complete("error: Failed to process screenshot: ${e.message}")
                        } finally {
                            // Restore overlay drawing state
                            if (hideOverlay) {
                                overlayManager.setDrawingEnabled(wasOverlayDrawingEnabled)
                            }
                        }
                    }

                    override fun onFailure(errorCode: Int) {
                        val errorMessage = when (errorCode) {
                            ERROR_TAKE_SCREENSHOT_INTERNAL_ERROR -> "Internal error occurred"
                            ERROR_TAKE_SCREENSHOT_INTERVAL_TIME_SHORT -> "Screenshot interval too short"
                            ERROR_TAKE_SCREENSHOT_INVALID_DISPLAY -> "Invalid display"
                            ERROR_TAKE_SCREENSHOT_NO_ACCESSIBILITY_ACCESS -> "No accessibility access"
                            ERROR_TAKE_SCREENSHOT_SECURE_WINDOW -> "Secure window cannot be captured"
                            else -> "Unknown error (code: $errorCode)"
                        }
                        Log.e(TAG, "Screenshot failed: $errorMessage")
                        future.complete("error: Screenshot failed: $errorMessage")

                        // Restore overlay drawing state
                        if (hideOverlay) {
                            overlayManager.setDrawingEnabled(wasOverlayDrawingEnabled)
                        }
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "Error taking screenshot", e)
            future.complete("error: Failed to take screenshot: ${e.message}")

            // Restore overlay drawing state in case of exception
            if (hideOverlay) {
                overlayManager.setDrawingEnabled(wasOverlayDrawingEnabled)
            }
        }
    }

    override fun onInterrupt() {
        Log.d(TAG, "Accessibility service interrupted")
        stopPeriodicUpdates()
        stopSocketServer()
        sendAccessibilityServiceStatusBroadcast(false)
    }

    private fun sendAccessibilityServiceStatusBroadcast(isEnabled: Boolean) {
        val intent = Intent("com.jeeves.ACCESSIBILITY_SERVICE_STATUS")
        intent.putExtra("is_enabled", isEnabled)
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopPeriodicUpdates()
        stopSocketServer()
        clearElementList()
        indexedElements.clear()
        configManager.removeListener(this)
        instance = null
        Log.d(TAG, "Accessibility service destroyed")
    }

    private fun addElementAndChildrenToOverlay(element: ElementNode, depth: Int) {
        // This method is now deprecated - we use the flat list approach in refreshVisibleElements
        // Keeping for compatibility but it's not used
        overlayManager.addElement(
            text = element.text,
            rect = element.rect,
            type = element.className,
            index = element.overlayIndex
        )

        for (child in element.children) {
            addElementAndChildrenToOverlay(child, depth + 1)
        }
    }

    /**
     * Calculate the coordinate system offset to fix mismatch between
     * getBoundsInScreen() (full display coordinates) and overlay drawing (app content coordinates)
     */
    private fun calculateCoordinateOffset(windowManager: WindowManager): Int {
        return try {
            val windowMetrics = windowManager.currentWindowMetrics
            val windowInsets = windowMetrics.windowInsets
            
            // Get system window insets (status bar, navigation bars, etc.)
            val systemBarsInsets = windowInsets.getInsets(WindowInsets.Type.systemBars())
            
            // The offset is the top system bar inset (status bar + any additional system UI)
            val calculatedOffset = systemBarsInsets.top
            
            Log.d(TAG, "Display bounds: ${windowMetrics.bounds}")
            Log.d(TAG, "System bars insets: top=${systemBarsInsets.top}, bottom=${systemBarsInsets.bottom}")
            Log.d(TAG, "Calculated coordinate offset: ${calculatedOffset}px")
            
            calculatedOffset
            
        } catch (e: Exception) {
            Log.e(TAG, "Error calculating coordinate offset: ${e.message}")
            0 // Fallback to 0 if there's an error
        }
    }
}
