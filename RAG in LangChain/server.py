#here we make as a server. this could be use for many clients, because @tools are used just for one application, but such a format (MCP) is common for everyone
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

@mcp.tool()
def get_weather(city: str) -> str:
    data = {
        "Vilnius": "5°C, cloudy",
        "London": "9°C, rainy",
        "Tokyo": "13°C, clear",
    }
    return data.get(city, f"No weather data for {city}")

if __name__ == "__main__":
    mcp.run()