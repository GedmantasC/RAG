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