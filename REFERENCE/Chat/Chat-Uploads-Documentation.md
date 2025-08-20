# Chat File Upload System Documentation

## Overview

The Chat File Upload System in tldw_chatbook provides a flexible, extensible mechanism for attaching and processing various file types in chat conversations. Originally supporting only images, the system now handles text files, code files, data files, and more through a plugin-based architecture.

## Table of Contents

1. [Architecture](#architecture)
2. [Supported File Types](#supported-file-types)
3. [User Guide](#user-guide)
4. [Developer Guide](#developer-guide)
5. [File Handler System](#file-handler-system)
6. [Implementation Details](#implementation-details)
7. [Extending the System](#extending-the-system)
8. [Migration Guide](#migration-guide)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

## Architecture

The file upload system consists of three main components:

```
┌─────────────────────────┐
│   UI Layer              │
│ (Chat_Window_Enhanced)  │
│  - File picker dialog   │
│  - Attachment indicator │
│  - Upload button        │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   File Handler Layer    │
│ (file_handlers.py)      │
│  - Handler registry     │
│  - Type-specific logic  │
│  - Content processing   │
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│   Event Handler Layer   │
│ (chat_events.py)        │
│  - Message creation     │
│  - API integration      │
│  - Database storage     │
└─────────────────────────┘
```

### Design Principles

1. **Backward Compatibility**: Existing image handling remains unchanged
2. **Extensibility**: Easy to add new file type handlers
3. **Type-Specific Processing**: Different file types are handled appropriately
4. **User Experience**: Clear feedback and intuitive behavior
5. **Security**: File validation and size limits

## Supported File Types

### Images (Attachments)
- **Extensions**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.tif`, `.svg`
- **Behavior**: Stored as binary attachments, sent with messages
- **Size Limit**: Configurable (default: varies by provider)
- **Processing**: Validated, potentially resized, base64 encoded

### Text Files (Inline)
- **Extensions**: `.txt`, `.md`, `.log`, `.text`, `.rst`, `.textile`
- **Behavior**: Content inserted directly into chat input
- **Size Limit**: 1MB
- **Processing**: UTF-8 decoding, wrapped with filename markers

### Code Files (Inline)
- **Extensions**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.h`, `.cs`, `.rb`, `.go`, `.rs`, etc.
- **Behavior**: Content inserted with syntax highlighting markers
- **Size Limit**: 512KB
- **Processing**: Language detection, code block formatting

### Data Files (Inline)
- **Extensions**: `.json`, `.yaml`, `.yml`, `.csv`, `.tsv`
- **Behavior**: Content formatted and inserted
- **Size Limit**: 256KB
- **Processing**: Validation, pretty-printing, CSV table formatting

### Other Files (Reference)
- **Extensions**: All others
- **Behavior**: File info inserted as reference
- **Processing**: Filename, size, and MIME type display

## User Guide

### Attaching Files

1. Click the 📎 (paperclip) button next to the chat input
2. Select your file from the file picker dialog
3. The file will be processed based on its type:
   - **Images**: Show as "📷 filename.jpg" indicator
   - **Text/Code/Data**: Content appears in chat input
   - **Others**: Reference info appears in chat input

### File Type Behaviors

#### Images
```
User clicks 📎 → Selects image.png → Indicator shows "📷 image.png"
→ User types message → Sends → Image attached to message
```

#### Text/Code Files
```
User clicks 📎 → Selects script.py → Content inserted:
```python
# File: script.py
def hello_world():
    print("Hello, World!")
```
→ User can edit/add context → Sends as regular message
```

#### Data Files
```
User clicks 📎 → Selects data.json → Formatted content inserted:
--- Data from data.json ---
{
  "name": "example",
  "value": 123
}
--- End of data.json ---
```

### Clearing Attachments

- For images: Click the ✓ indicator or send the message
- For inserted text: Edit or clear the chat input normally

## Developer Guide

### File Handler System

The system uses a plugin architecture where each file type has a dedicated handler:

```python
from abc import ABC, abstractmethod

class FileHandler(ABC):
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the file."""
        pass
    
    @abstractmethod
    async def process(self, file_path: Path) -> ProcessedFile:
        """Process the file and return result."""
        pass
```

### ProcessedFile Structure

```python
@dataclass
class ProcessedFile:
    content: Optional[str] = None  # For inline insertion
    attachment_data: Optional[bytes] = None  # For binary attachments
    attachment_mime_type: Optional[str] = None  # MIME type
    display_name: str = ""  # UI display name
    insert_mode: Literal["inline", "attachment"] = "inline"
    file_type: str = "unknown"  # Type identifier
```

### Handler Registration

Handlers are registered in priority order:

```python
class FileHandlerRegistry:
    def __init__(self):
        self.handlers = [
            ImageFileHandler(),      # Highest priority
            TextFileHandler(),
            CodeFileHandler(),
            DataFileHandler(),
            DefaultFileHandler(),    # Lowest priority (catch-all)
        ]
```

## Implementation Details

### UI Integration

The enhanced chat window maintains both old and new attachment systems:

```python
class ChatWindowEnhanced(Container):
    pending_image = reactive(None)  # Legacy support
    pending_attachment = None       # New system
```

### Event Flow

1. **File Selection**
   ```python
   handle_attach_image_button() → FileOpen dialog → on_file_selected()
   ```

2. **File Processing**
   ```python
   process_file_attachment() → file_handler_registry.process_file()
   → Handler.process() → ProcessedFile
   ```

3. **Content Handling**
   ```python
   if processed_file.insert_mode == "inline":
       # Insert into chat input
       chat_input.text += processed_file.content
   else:
       # Store as attachment
       self.pending_attachment = {...}
   ```

4. **Message Sending**
   ```python
   handle_chat_send_button_pressed() → Check pending_attachment
   → Include in API call if present → Clear after sending
   ```

### Database Storage

Images and binary attachments are stored in the messages table:

```sql
-- Existing schema supports image storage
messages (
    ...
    image_data BLOB,
    image_mime_type TEXT,
    ...
)
```

## Extending the System

### Adding a New File Handler

1. **Create Handler Class**
   ```python
   class PDFFileHandler(FileHandler):
       SUPPORTED_EXTENSIONS = {'.pdf'}
       
       def can_handle(self, file_path: Path) -> bool:
           return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
       
       async def process(self, file_path: Path) -> ProcessedFile:
           # Extract text from PDF
           text = await extract_pdf_text(file_path)
           return ProcessedFile(
               content=f"--- PDF: {file_path.name} ---\n{text}\n---",
               display_name=file_path.name,
               insert_mode="inline",
               file_type="document"
           )
   ```

2. **Register Handler**
   ```python
   # In FileHandlerRegistry.__init__()
   self.handlers = [
       ImageFileHandler(),
       PDFFileHandler(),  # Add here
       TextFileHandler(),
       # ...
   ]
   ```

3. **Update File Filters** (optional)
   ```python
   # In handle_attach_image_button()
   file_filters = Filters(
       ("All Supported", "*.png;*.jpg;...;*.pdf"),
       ("Documents", "*.pdf;*.doc;*.docx"),
       # ...
   )
   ```

### Adding New Insert Modes

To add a new insertion mode (e.g., "sidebar"):

1. Update `ProcessedFile`:
   ```python
   insert_mode: Literal["inline", "attachment", "sidebar"] = "inline"
   ```

2. Handle in `process_file_attachment()`:
   ```python
   elif processed_file.insert_mode == "sidebar":
       # Custom handling for sidebar display
       self.show_in_sidebar(processed_file)
   ```

## Migration Guide

### From Image-Only to Multi-File

The system maintains full backward compatibility:

1. **Existing Code**
   ```python
   # Old way still works
   pending_image = chat_window.get_pending_image()
   if pending_image:
       # Process image
   ```

2. **New Code**
   ```python
   # New way with fallback
   if chat_window.pending_attachment:
       # Handle any attachment type
   elif chat_window.pending_image:
       # Legacy image handling
   ```

### Configuration

Enable/disable file uploads in config:

```toml
[chat.images]
show_attach_button = true  # Controls upload button visibility

[chat.uploads]
max_text_file_size = 1048576  # 1MB in bytes
max_code_file_size = 524288   # 512KB
max_data_file_size = 262144   # 256KB
allowed_extensions = [
    ".txt", ".md", ".py", ".js", 
    # ... add more as needed
]
```

## API Reference

### ChatWindowEnhanced Methods

```python
async def process_file_attachment(self, file_path: str) -> None:
    """Process any file type through the handler system."""

def get_pending_image(self) -> Optional[dict]:
    """Legacy method for image attachments."""

async def toggle_attach_button_visibility(self, show: bool) -> None:
    """Show/hide the attach button dynamically."""
```

### FileHandler Methods

```python
def can_handle(self, file_path: Path) -> bool:
    """Returns True if handler can process this file type."""

async def process(self, file_path: Path) -> ProcessedFile:
    """Process file and return ProcessedFile result."""

def get_mime_type(self, file_path: Path) -> str:
    """Get MIME type for file (utility method)."""
```

### FileHandlerRegistry Methods

```python
async def process_file(self, file_path: Union[str, Path]) -> ProcessedFile:
    """Route file to appropriate handler and process it."""
```

## Troubleshooting

### Common Issues

1. **File Not Processing**
   - Check file size limits
   - Verify file extension is supported
   - Check file permissions

2. **Content Not Inserting**
   - Ensure chat input is not at maximum length
   - Check for encoding issues with non-UTF8 files

3. **Images Not Attaching**
   - Verify image format is supported
   - Check image size (provider limits)
   - Ensure vision-capable model is selected

### Debug Logging

Enable debug logging to troubleshoot:

```python
# Check logs for:
"Using {HandlerName} for {filename}"
"Pending attachment: {type} ({name})"
"Including image in API call"
```

### Error Messages

| Error | Meaning | Solution |
|-------|---------|----------|
| "File too large" | Exceeds size limit | Use smaller file or increase limit |
| "Invalid JSON/YAML" | Data file malformed | Fix file syntax |
| "No handler for file" | Unsupported type | Add custom handler |
| "Permission denied" | Can't read file | Check file permissions |

## Security Considerations

1. **File Size Limits**: Prevent memory exhaustion
2. **Path Validation**: No directory traversal
3. **Content Validation**: JSON/YAML parsing in safe mode
4. **Binary Validation**: Image format verification
5. **Encoding Safety**: Handle non-UTF8 gracefully

## Future Enhancements

Potential improvements to the system:

1. **Drag & Drop**: Direct file dropping into chat
2. **Multiple Files**: Batch file selection
3. **Preview**: Show file previews before sending
4. **Compression**: Auto-compress large files
5. **URL Support**: Fetch and attach from URLs
6. **Cloud Storage**: Direct integration with cloud services
7. **Audio/Video**: Transcription support
8. **Documents**: PDF, DOCX text extraction
9. **Archives**: ZIP/TAR content listing
10. **Streaming**: Progressive file processing

## Conclusion

The Chat File Upload System provides a robust, extensible foundation for handling diverse file types in chat conversations. Its plugin architecture ensures easy maintenance and feature additions while maintaining backward compatibility with existing functionality.

For questions or contributions, please refer to the main project documentation or submit an issue on the repository.