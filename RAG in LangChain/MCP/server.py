#this is different form @tool that it can be accesed by any ai agent. @tool is a way to call local function, MCP can be called by anyone. So @tool is used localy, @MCP could be used by anyone who has an access
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