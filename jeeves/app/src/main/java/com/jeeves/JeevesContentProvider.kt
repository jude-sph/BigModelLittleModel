package com.jeeves

import android.content.ContentProvider
import android.content.ContentValues
import android.content.UriMatcher
import android.content.pm.ApplicationInfo
import android.content.pm.PackageInfo
import android.database.Cursor
import android.database.MatrixCursor
import android.graphics.Rect
import android.net.Uri
import android.os.Build
import android.util.Log
import android.view.accessibility.AccessibilityNodeInfo
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import androidx.core.net.toUri
import android.os.Bundle
import com.jeeves.model.ElementNode
import com.jeeves.model.PhoneState
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ResolveInfo

class JeevesContentProvider : ContentProvider() {
    companion object {
        private const val TAG = "JeevesContentProvider"
        private const val AUTHORITY = "com.jeeves"
        private const val A11Y_TREE = 1
        private const val PHONE_STATE = 2
        private const val PING = 3
        private const val KEYBOARD_ACTIONS = 4
        private const val STATE = 5
        private const val OVERLAY_OFFSET = 6
        private const val PACKAGES = 7
        private const val TAP_BY_INDEX = 8 // New endpoint for tapping
        private const val BOUNDING_BOXES = 9 // Endpoint for bounding box data

        private val uriMatcher = UriMatcher(UriMatcher.NO_MATCH).apply {
            addURI(AUTHORITY, "a11y_tree", A11Y_TREE)
            addURI(AUTHORITY, "phone_state", PHONE_STATE)
            addURI(AUTHORITY, "ping", PING)
            addURI(AUTHORITY, "keyboard/*", KEYBOARD_ACTIONS)
            addURI(AUTHORITY, "state", STATE)
            addURI(AUTHORITY, "overlay_offset", OVERLAY_OFFSET)
            addURI(AUTHORITY, "packages", PACKAGES)
            addURI(AUTHORITY, "tap_by_index", TAP_BY_INDEX) // New tap endpoint
            addURI(AUTHORITY, "bounding_boxes", BOUNDING_BOXES) // Bounding boxes endpoint
        }
    }

    override fun onCreate(): Boolean {
        Log.d(TAG, "JeevesContentProvider created")
        return true
    }

    override fun query(
        uri: Uri,
        projection: Array<String>?,
        selection: String?,
        selectionArgs: Array<String>?,
        sortOrder: String?
    ): Cursor? {
        val cursor = MatrixCursor(arrayOf("result"))

        try {
            val result = when (uriMatcher.match(uri)) {
                A11Y_TREE -> getAccessibilityTree()
                PHONE_STATE -> getPhoneState()
                PING -> createSuccessResponse("pong")
                STATE -> getCombinedState()
                PACKAGES -> getInstalledPackagesJson()
                BOUNDING_BOXES -> getBoundingBoxes()
                else -> createErrorResponse("Unknown endpoint: ${uri.path}")
            }

            cursor.addRow(arrayOf(result))

        } catch (e: Exception) {
            Log.e(TAG, "Query execution failed", e)
            cursor.addRow(arrayOf(createErrorResponse("Execution failed: ${e.message}")))
        }

        return cursor
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? {
        return when (uriMatcher.match(uri)) {
            KEYBOARD_ACTIONS -> executeKeyboardAction(uri, values)
            OVERLAY_OFFSET -> updateOverlayOffset(uri, values)
            TAP_BY_INDEX -> performTapByIndex(values)
            else -> "content://$AUTHORITY/result?status=error&message=${Uri.encode("Unsupported insert endpoint: ${uri.path}")}".toUri()
        }
    }

    private fun executeKeyboardAction(uri: Uri, values: ContentValues?): Uri? {
        if (values == null) {
            return "content://$AUTHORITY/result?status=error&message=No values provided".toUri()
        }

        try {
            val action = uri.lastPathSegment ?: return "content://$AUTHORITY/result?status=error&message=No action specified".toUri()

            val result = when (action) {
                "input" -> performKeyboardInputBase64(values)
                "clear" -> performKeyboardClear()
                "key" -> performKeyboardKey(values)
                else -> "error: Unknown keyboard action: $action"
            }

            // Encode result in URI
            return if (result.startsWith("success")) {
                "content://$AUTHORITY/result?status=success&message=${Uri.encode(result)}".toUri()
            } else {
                "content://$AUTHORITY/result?status=error&message=${Uri.encode(result)}".toUri()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Keyboard action execution failed", e)
            return "content://$AUTHORITY/result?status=error&message=${Uri.encode("Execution failed: ${e.message}")}".toUri()
        }
    }

    private fun updateOverlayOffset(uri: Uri, values: ContentValues?): Uri? {
        if (values == null) {
            return "content://$AUTHORITY/result?status=error&message=No values provided".toUri()
        }

        try {
            val accessibilityService = JeevesAccessibilityService.getInstance()
                ?: return "content://$AUTHORITY/result?status=error&message=Accessibility service not available".toUri()

            var messages = mutableListOf<String>()
            
            // Handle overlay visibility if provided
            val visible = values.getAsBoolean("visible")
            if (visible != null) {
                val visSuccess = accessibilityService.setOverlayVisible(visible)
                if (visSuccess) {
                    messages.add("Overlay visibility set to $visible")
                } else {
                    return "content://$AUTHORITY/result?status=error&message=Failed to set overlay visibility".toUri()
                }
            }
            
            // Handle offset if provided
            val offset = values.getAsInteger("offset")
            if (offset != null) {
                val offsetSuccess = accessibilityService.setOverlayOffset(offset)
                if (offsetSuccess) {
                    messages.add("Overlay offset updated to $offset")
                } else {
                    return "content://$AUTHORITY/result?status=error&message=Failed to update overlay offset".toUri()
                }
            }

            // If neither parameter was provided
            if (messages.isEmpty()) {
                return "content://$AUTHORITY/result?status=error&message=No valid parameters provided (visible or offset)".toUri()
            }

            return "content://$AUTHORITY/result?status=success&message=${Uri.encode(messages.joinToString(", "))}".toUri()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to update overlay settings", e)
            return "content://$AUTHORITY/result?status=error&message=${Uri.encode("Execution failed: ${e.message}")}".toUri()
        }
    }

    private fun performTapByIndex(values: ContentValues?): Uri? {
        if (values == null) {
            return "content://$AUTHORITY/result?status=error&message=No values provided".toUri()
        }

        try {
            val accessibilityService = JeevesAccessibilityService.getInstance()
                ?: return "content://$AUTHORITY/result?status=error&message=Accessibility service not available".toUri()

            val index = values.getAsInteger("index")
                ?: return "content://$AUTHORITY/result?status=error&message=No index provided".toUri()

            val result = accessibilityService.performTapByIndex(index)
            
            return if (result.startsWith("success")) {
                "content://$AUTHORITY/result?status=success&message=${Uri.encode(result)}".toUri()
            } else {
                "content://$AUTHORITY/result?status=error&message=${Uri.encode(result)}".toUri()
            }

        } catch (e: Exception) {
            Log.e(TAG, "Failed to perform tap by index", e)
            return "content://$AUTHORITY/result?status=error&message=${Uri.encode("Tap failed: ${e.message}")}".toUri()
        }
    }

    private fun getAccessibilityTree(): String {
        val accessibilityService = JeevesAccessibilityService.getInstance()
            ?: return createErrorResponse("Accessibility service not available")
        return try {

            val treeJson = accessibilityService.getVisibleElements().map { element ->
                buildElementNodeJson(element)
            }

            createSuccessResponse(treeJson.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get accessibility tree", e)
            createErrorResponse("Failed to get accessibility tree: ${e.message}")
        }
    }

    private fun buildElementNodeJson(element: ElementNode): JSONObject {
        return JSONObject().apply {
            put("index", element.overlayIndex)
            put("resourceId", element.nodeInfo.viewIdResourceName ?: "")
            put("className", element.className)
            put("text", element.text)
            put("bounds", "${element.rect.left}, ${element.rect.top}, ${element.rect.right}, ${element.rect.bottom}")

            // Recursively build children JSON
            val childrenArray = org.json.JSONArray()
            element.children.forEach { child ->
                childrenArray.put(buildElementNodeJson(child))
            }
            put("children", childrenArray)
        }
    }


    private fun getPhoneState(): String {
        val accessibilityService = JeevesAccessibilityService.getInstance()
            ?: return createErrorResponse("Accessibility service not available")
        return try {
            val phoneState = buildPhoneStateJson(accessibilityService.getPhoneState())
            createSuccessResponse(phoneState.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get accessibility tree", e)
            createErrorResponse("Failed to get accessibility tree: ${e.message}")
        }
    }

    private fun buildPhoneStateJson(phoneState: PhoneState) =
        JSONObject().apply {
            put("currentApp", phoneState.appName)
            put("packageName", phoneState.packageName)
            put("keyboardVisible", phoneState.keyboardVisible)
            put("focusedElement", JSONObject().apply {
                val rect = Rect()
                put("text", phoneState.focusedElement?.text)
                put("className", phoneState.focusedElement?.className)
                put("resourceId", phoneState.focusedElement?.viewIdResourceName ?: "")
            })
        }

    private fun getCombinedState(): String {
        val accessibilityService = JeevesAccessibilityService.getInstance()
            ?: return createErrorResponse("Accessibility service not available")

        return try {
            // Get accessibility tree
            val treeJson = accessibilityService.getVisibleElements().map { element ->
                buildElementNodeJson(element)
            }

            // Get phone state
            val phoneStateJson = buildPhoneStateJson(accessibilityService.getPhoneState())

            // Combine both in a single response
            val combinedState = JSONObject().apply {
                put("a11y_tree", org.json.JSONArray(treeJson))
                put("phone_state", phoneStateJson)
            }

            createSuccessResponse(combinedState.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get combined state", e)
            createErrorResponse("Failed to get combined state: ${e.message}")
        }
    }

    private fun performTextInput(values: ContentValues): String {
        val accessibilityService = JeevesAccessibilityService.getInstance()
            ?: return "error: Accessibility service not available"
        // Get the hex-encoded text
        val hexText = values.getAsString("hex_text")
            ?: return "error: No hex_text provided"

        // Check if we should append (default is false = replace)
        val append = values.getAsBoolean("append") ?: false

        // Decode hex to actual text
        val text = try {
            hexText.chunked(2).map { it.toInt(16).toChar() }.joinToString("")
        } catch (e: Exception) {
            return "error: Invalid hex encoding: ${e.message}"
        }

        // Find the currently focused element
        val focusedNode = accessibilityService.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            ?: return "error: No focused input element found"

        return try {
            val finalText = if (append) {
                // Get existing text and append to it
                val existingText = focusedNode.text?.toString() ?: ""
                existingText + text
            } else {
                // Just use the new text (replace)
                text
            }

            // Set the text using ACTION_SET_TEXT
            val arguments = Bundle().apply {
                putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, finalText)
            }
            val result = focusedNode.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, arguments)
            focusedNode.recycle()

            if (result) {
                val mode = if (append) "appended" else "set"
                "success: Text $mode - '$text'"
            } else {
                "error: Text input failed"
            }
        } catch (e: Exception) {
            focusedNode.recycle()
            "error: Text input exception: ${e.message}"
        }
    }

    private fun performKeyboardInputBase64(values: ContentValues): String {
        val base64Text = values.getAsString("base64_text") ?: return "error: no text provided"
        val append = values.getAsBoolean("append") ?: false

        return if (JeevesKeyboardIME.getInstance() != null) {
            val ok = JeevesKeyboardIME.getInstance()!!.inputB64Text(base64Text, append)
            if (ok) "success: input done (append=$append)" else "error: input failed"
        } else {
            "error: IME not active"
        }
    }


    private fun performKeyboardClear(): String {
        val keyboardIME = JeevesKeyboardIME.getInstance()
            ?: return "error: JeevesKeyboardIME not active or available"

        if (!keyboardIME.hasInputConnection()) {
            return "error: No input connection available - keyboard may not be focused on an input field"
        }

        return if (keyboardIME.clearText()) {
            "success: Text cleared via keyboard"
        } else {
            "error: Failed to clear text via keyboard"
        }
    }

    private fun performKeyboardKey(values: ContentValues): String {
        val keyboardIME = JeevesKeyboardIME.getInstance()
            ?: return "error: JeevesKeyboardIME not active or available"

        if (!keyboardIME.hasInputConnection()) {
            return "error: No input connection available - keyboard may not be focused on an input field"
        }

        val keyCode = values.getAsInteger("key_code")
            ?: return "error: No key_code provided"

        return if (keyboardIME.sendKeyEventDirect(keyCode)) {
            "success: Key event sent via keyboard - code: $keyCode"
        } else {
            "error: Failed to send key event via keyboard"
        }
    }


    private fun getInstalledPackagesJson(): String {
        val pm = context?.packageManager ?: return createErrorResponse("PackageManager unavailable")

        return try {
            val mainIntent = Intent(Intent.ACTION_MAIN, null).apply {
                addCategory(Intent.CATEGORY_LAUNCHER)
            }

            val resolvedApps: List<ResolveInfo> = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                pm.queryIntentActivities(mainIntent, PackageManager.ResolveInfoFlags.of(0L))
            } else {
                @Suppress("DEPRECATION")
                pm.queryIntentActivities(mainIntent, 0)
            }

            val arr = JSONArray()

            for (resolveInfo in resolvedApps) {
                val pkgInfo = try {
                    pm.getPackageInfo(resolveInfo.activityInfo.packageName, 0)
                } catch (e: PackageManager.NameNotFoundException) {
                    continue
                }

                val appInfo = resolveInfo.activityInfo.applicationInfo
                val obj = JSONObject()

                obj.put("packageName", pkgInfo.packageName)
                obj.put("label", resolveInfo.loadLabel(pm).toString())
                obj.put("versionName", pkgInfo.versionName ?: JSONObject.NULL)

                val versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    pkgInfo.longVersionCode
                } else {
                    @Suppress("DEPRECATION")
                    pkgInfo.versionCode.toLong()
                }
                obj.put("versionCode", versionCode)

                val isSystem = (appInfo.flags and ApplicationInfo.FLAG_SYSTEM) != 0
                obj.put("isSystemApp", isSystem)

                arr.put(obj)
            }

            val root = JSONObject()
            root.put("status", "success")
            root.put("count", arr.length())
            root.put("packages", arr)

            root.toString()

        } catch (e: Exception) {
            Log.e(TAG, "Failed to enumerate launchable apps", e)
            createErrorResponse("Failed to enumerate launchable apps: ${e.message}")
        }
    }

    private fun getBoundingBoxes(): String {
        val accessibilityService = JeevesAccessibilityService.getInstance()
            ?: return createErrorResponse("Accessibility service not available")

        return try {
            val elements = accessibilityService.getVisibleElements()
            val boxesArray = JSONArray()

            // Flatten all elements including children
            fun addElementAndChildren(element: ElementNode) {
                val box = JSONObject().apply {
                    put("index", element.overlayIndex)
                    put("left", element.rect.left)
                    put("top", element.rect.top)
                    put("right", element.rect.right)
                    put("bottom", element.rect.bottom)
                    put("width", element.rect.width())
                    put("height", element.rect.height())
                    put("text", element.text)
                    put("type", element.className)
                }
                boxesArray.put(box)

                // Recursively add children
                element.children.forEach { child ->
                    addElementAndChildren(child)
                }
            }

            elements.forEach { root ->
                addElementAndChildren(root)
            }

            val result = JSONObject().apply {
                put("boxes", boxesArray)
                put("count", boxesArray.length())
            }

            createSuccessResponse(result.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get bounding boxes", e)
            createErrorResponse("Failed to get bounding boxes: ${e.message}")
        }
    }

    private fun createSuccessResponse(data: String): String {
        return JSONObject().apply {
            put("status", "success")
            put("message", data)
        }.toString()
    }

    private fun createErrorResponse(error: String): String {
        return JSONObject().apply {
            put("status", "error")
            put("message", error)
        }.toString()
    }

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<String>?): Int = 0
    override fun update(uri: Uri, values: ContentValues?, selection: String?, selectionArgs: Array<String>?): Int = 0
    override fun getType(uri: Uri): String? = null
}
