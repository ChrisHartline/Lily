"""
Clara Tools - Tool registry and implementations

Tools are functions that Clara can invoke to perform actions.
Each tool has:
- name: Unique identifier
- description: What the tool does (for routing/display)
- parameters: JSON schema of expected inputs
- handler: The actual implementation
"""

import os
import json
import base64
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import httpx


@dataclass
class ToolDefinition:
    """Definition of a tool that Clara can use"""
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Optional[Callable] = None
    category: str = "general"
    requires_confirmation: bool = False

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "category": self.category,
            "requires_confirmation": self.requires_confirmation
        }


@dataclass
class ToolResult:
    """Result from a tool execution"""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata
        }


class ToolRegistry:
    """
    Registry for Clara's tools.

    Manages tool registration, discovery, and execution.
    """

    def __init__(self, debug: bool = False):
        self.tools: Dict[str, ToolDefinition] = {}
        self.debug = debug
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """Register the built-in tools"""

        # Summarize tool
        self.register(ToolDefinition(
            name="summarize",
            description="Summarize text or documents into key points",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize"
                    },
                    "max_points": {
                        "type": "integer",
                        "description": "Maximum number of bullet points",
                        "default": 5
                    },
                    "style": {
                        "type": "string",
                        "enum": ["bullets", "paragraph", "tldr"],
                        "default": "bullets"
                    }
                },
                "required": ["text"]
            },
            handler=self._handle_summarize,
            category="ai"
        ))

        # Text-to-speech tool
        self.register(ToolDefinition(
            name="text_to_speech",
            description="Convert text to spoken audio",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to speak"
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code (e.g., 'en', 'es')",
                        "default": "en"
                    }
                },
                "required": ["text"]
            },
            handler=self._handle_tts,
            category="audio"
        ))

        # File read tool
        self.register(ToolDefinition(
            name="file_read",
            description="Read and extract content from uploaded files",
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "Extract text content",
                        "default": True
                    }
                },
                "required": ["file_path"]
            },
            handler=self._handle_file_read,
            category="file"
        ))

        # Web search tool
        self.register(ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_web_search,
            category="web"
        ))

        # Get current time/date
        self.register(ToolDefinition(
            name="get_datetime",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., 'UTC', 'America/New_York')",
                        "default": "UTC"
                    },
                    "format": {
                        "type": "string",
                        "description": "Output format",
                        "default": "human"
                    }
                }
            },
            handler=self._handle_datetime,
            category="utility"
        ))

        # Memory recall tool (for explicit memory queries)
        self.register(ToolDefinition(
            name="memory_recall",
            description="Search Clara's memory for past conversations",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in memory"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 5
                    }
                },
                "required": ["query"]
            },
            handler=None,  # Handled by Clara directly
            category="memory"
        ))

    def register(self, tool: ToolDefinition):
        """Register a tool"""
        self.tools[tool.name] = tool
        if self.debug:
            print(f"   [Tools] Registered: {tool.name}")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name"""
        return self.tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Dict]:
        """List all available tools"""
        tools = self.tools.values()
        if category:
            tools = [t for t in tools if t.category == category]
        return [t.to_dict() for t in tools]

    def get_categories(self) -> List[str]:
        """Get all tool categories"""
        return list(set(t.category for t in self.tools.values()))

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given arguments"""
        tool = self.get(name)

        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: {name}"
            )

        if tool.handler is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"Tool {name} has no handler (handled externally)"
            )

        try:
            if self.debug:
                print(f"   [Tools] Executing: {name}({arguments})")

            # Call handler (may be sync or async)
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(arguments)
            else:
                result = tool.handler(arguments)

            return ToolResult(
                tool_name=name,
                success=True,
                result=result
            )

        except Exception as e:
            return ToolResult(
                tool_name=name,
                success=False,
                error=str(e)
            )

    # === Tool Handlers ===

    def _handle_summarize(self, args: Dict) -> Dict:
        """
        Summarize text.

        In production, this would call an LLM. For now, we return a placeholder
        that gets enhanced by Clara's response.
        """
        text = args.get("text", "")
        max_points = args.get("max_points", 5)
        style = args.get("style", "bullets")

        # Simple extractive summary (first N sentences)
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        summary_sentences = sentences[:max_points]

        if style == "bullets":
            result = "\n".join(f"• {s}" for s in summary_sentences)
        elif style == "tldr":
            result = f"TL;DR: {summary_sentences[0]}" if summary_sentences else "No content to summarize."
        else:
            result = " ".join(summary_sentences)

        return {
            "summary": result,
            "original_length": len(text),
            "summary_length": len(result),
            "num_points": len(summary_sentences)
        }

    def _handle_tts(self, args: Dict) -> Dict:
        """Convert text to speech using gTTS"""
        text = args.get("text", "")
        language = args.get("language", "en")

        try:
            from gtts import gTTS
            import io

            # Generate speech
            tts = gTTS(text=text, lang=language)

            # Save to bytes
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)

            # Encode as base64 for transmission
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')

            return {
                "audio_base64": audio_base64,
                "format": "mp3",
                "language": language,
                "text_length": len(text)
            }

        except ImportError:
            return {
                "error": "gTTS not installed. Install with: pip install gtts",
                "text": text
            }
        except Exception as e:
            return {
                "error": str(e),
                "text": text
            }

    def _handle_file_read(self, args: Dict) -> Dict:
        """Read file contents"""
        file_path = args.get("file_path", "")
        extract_text = args.get("extract_text", True)

        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}"}

        try:
            # Get file info
            stat = os.stat(file_path)
            file_info = {
                "path": file_path,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            }

            if extract_text:
                # Simple text extraction
                ext = os.path.splitext(file_path)[1].lower()

                if ext in ['.txt', '.md', '.py', '.js', '.ts', '.json', '.yaml', '.yml']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    file_info["content"] = content[:10000]  # Limit content size
                    file_info["truncated"] = len(content) > 10000
                else:
                    file_info["content"] = f"[Binary file: {ext}]"
                    file_info["extract_supported"] = False

            return file_info

        except Exception as e:
            return {"error": str(e), "path": file_path}

    async def _handle_web_search(self, args: Dict) -> Dict:
        """
        Web search placeholder.

        In production, this would call a search API (DuckDuckGo, Brave, etc.)
        """
        query = args.get("query", "")
        num_results = args.get("num_results", 5)

        # Placeholder - in production use a real search API
        return {
            "query": query,
            "results": [],
            "message": "Web search requires API integration. Configure a search provider.",
            "suggested_providers": ["DuckDuckGo API", "Brave Search API", "SerpAPI"]
        }

    def _handle_datetime(self, args: Dict) -> Dict:
        """Get current datetime"""
        from datetime import timezone

        tz_str = args.get("timezone", "UTC")
        fmt = args.get("format", "human")

        now = datetime.now(timezone.utc)

        if fmt == "iso":
            formatted = now.isoformat()
        elif fmt == "unix":
            formatted = int(now.timestamp())
        else:  # human
            formatted = now.strftime("%A, %B %d, %Y at %I:%M %p UTC")

        return {
            "datetime": formatted,
            "timezone": tz_str,
            "format": fmt
        }


# Singleton instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry(debug: bool = False) -> ToolRegistry:
    """Get or create the global tool registry"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry(debug=debug)
    return _registry
