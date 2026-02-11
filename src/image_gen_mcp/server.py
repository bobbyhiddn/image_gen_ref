"""
MCP Server for image generation and revision using Gemini.
"""
import base64
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent

from .core import generate_image, load_dotenv, DEFAULT_MODEL


# Load environment variables
load_dotenv()

server = Server("image-gen")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available image generation tools."""
    return [
        Tool(
            name="generate_image",
            description="Generate a new image from a text description using Gemini AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save the image. If not provided, auto-generates a timestamped filename."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_MODEL})",
                        "default": DEFAULT_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="revise_image",
            description="Revise/edit an existing image based on text instructions using Gemini AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Instructions describing the changes to make to the image"
                    },
                    "input_image_path": {
                        "type": "string",
                        "description": "Path to the image file to revise. Required if input_image_base64 is not provided."
                    },
                    "input_image_base64": {
                        "type": "string",
                        "description": "Base64-encoded image data to revise. Required if input_image_path is not provided."
                    },
                    "input_image_mime_type": {
                        "type": "string",
                        "description": "MIME type of the base64 image (e.g., 'image/png'). Required when using input_image_base64.",
                        "default": "image/png"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save the revised image. If not provided, auto-generates a timestamped filename."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_MODEL})",
                        "default": DEFAULT_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="style_generate",
            description="Generate a new image in the style of a reference image. Provide a text prompt describing the image content and a path to a style reference image — the generated image will adopt the visual style, color palette, and aesthetic of the reference.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image content to generate"
                    },
                    "style_image_path": {
                        "type": "string",
                        "description": "Path to the style reference image whose visual style will be applied"
                    },
                    "style_image_base64": {
                        "type": "string",
                        "description": "Base64-encoded style reference image data. Alternative to style_image_path."
                    },
                    "style_image_mime_type": {
                        "type": "string",
                        "description": "MIME type of the base64 style image (e.g., 'image/png'). Required when using style_image_base64.",
                        "default": "image/png"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional file path to save the generated image. If not provided, auto-generates a timestamped filename."
                    },
                    "model": {
                        "type": "string",
                        "description": f"Gemini model ID to use (default: {DEFAULT_MODEL})",
                        "default": DEFAULT_MODEL
                    }
                },
                "required": ["prompt"]
            }
        ),
        Tool(
            name="list_models",
            description="List available Gemini models for image generation",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent | ImageContent]:
    """Handle tool calls."""

    if name == "list_models":
        models_info = f"""Available Gemini models for image generation:

- {DEFAULT_MODEL} (default)
  Latest image generation model

- gemini-2.0-flash-exp-image-generation
  Fast experimental image generation

- gemini-2.5-flash-preview-05-20
  Flash model with image capabilities

- gemini-2.5-pro-exp-03-25
  Pro model with enhanced capabilities

Set GEMINI_API_KEY environment variable to use these models."""
        return [TextContent(type="text", text=models_info)]

    elif name == "generate_image":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        output_path = arguments.get("output_path")
        model = arguments.get("model", DEFAULT_MODEL)

        try:
            result_path = generate_image(
                prompt=prompt,
                out_path=output_path,
                model=model,
            )

            # Read the generated image to return as ImageContent
            with open(result_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(result_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            return [
                TextContent(type="text", text=f"Generated image saved to: {result_path}"),
                ImageContent(type="image", data=image_data, mimeType=mime_type)
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating image: {e}")]

    elif name == "revise_image":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        input_image_path = arguments.get("input_image_path")
        input_image_base64 = arguments.get("input_image_base64")
        input_image_mime_type = arguments.get("input_image_mime_type", "image/png")
        output_path = arguments.get("output_path")
        model = arguments.get("model", DEFAULT_MODEL)

        if not input_image_path and not input_image_base64:
            return [TextContent(type="text", text="Error: Either 'input_image_path' or 'input_image_base64' is required")]

        try:
            input_bytes = None
            if input_image_base64:
                input_bytes = base64.b64decode(input_image_base64)

            result_path = generate_image(
                prompt=prompt,
                out_path=output_path,
                input_image_path=input_image_path,
                input_image_bytes=input_bytes,
                input_image_mime_type=input_image_mime_type,
                model=model,
            )

            with open(result_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(result_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            return [
                TextContent(type="text", text=f"Revised image saved to: {result_path}"),
                ImageContent(type="image", data=image_data, mimeType=mime_type)
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error revising image: {e}")]

    elif name == "style_generate":
        prompt = arguments.get("prompt")
        if not prompt:
            return [TextContent(type="text", text="Error: 'prompt' is required")]

        style_image_path = arguments.get("style_image_path")
        style_image_base64 = arguments.get("style_image_base64")
        style_image_mime_type = arguments.get("style_image_mime_type", "image/png")
        output_path = arguments.get("output_path")
        model = arguments.get("model", DEFAULT_MODEL)

        if not style_image_path and not style_image_base64:
            return [TextContent(type="text", text="Error: Either 'style_image_path' or 'style_image_base64' is required for style reference")]

        try:
            style_bytes = None
            if style_image_base64:
                style_bytes = base64.b64decode(style_image_base64)

            result_path = generate_image(
                prompt=prompt,
                out_path=output_path,
                style_ref_image_path=style_image_path,
                style_ref_image_bytes=style_bytes,
                style_ref_mime_type=style_image_mime_type,
                model=model,
            )

            with open(result_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            ext = os.path.splitext(result_path)[1].lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(ext, "image/png")

            return [
                TextContent(type="text", text=f"Style-generated image saved to: {result_path}"),
                ImageContent(type="image", data=image_data, mimeType=mime_type)
            ]
        except Exception as e:
            return [TextContent(type="text", text=f"Error generating styled image: {e}")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def run_server():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main():
    """Entry point for the MCP server."""
    import asyncio
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
