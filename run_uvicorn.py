"""
Run the ASGI server with uvicorn (this is a convenience script to run in PyCharm debugger).
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("versatilis.asgi:application")
