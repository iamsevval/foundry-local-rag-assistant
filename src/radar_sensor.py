import requests
import time

def get_live_radar_data():
    """
    Fetches live flight telemetry from the OpenSky Network ADS-B API.
    We bound the box around the Marmara/Istanbul region to find active flights.
    """
    # Bounding box for Istanbul region
    lamin, lomin, lamax, lomax = 40.7, 28.4, 41.3, 29.5
    url = f"https://opensky-network.org/api/states/all?lamin={lamin}&lomin={lomin}&lamax={lamax}&lomax={lomax}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and data.get("states") and len(data["states"]) > 0:
                # Pick the first aircraft found in the radar
                target = data["states"][0]
                
                # Extract telemetry (OpenSky structure)
                callsign = str(target[1]).strip()
                country = target[2]
                longitude = target[5]
                latitude = target[6]
                altitude = target[7] # baro altitude in meters
                velocity = target[9] # m/s
                heading = target[10] # degrees
                
                # Return structured telemetry
                return {
                    "status": "active",
                    "callsign": callsign or "UNKNOWN",
                    "country": country,
                    "altitude_m": altitude or 0,
                    "velocity_ms": velocity or 0,
                    "heading_deg": heading or 0,
                    "lat": latitude,
                    "lon": longitude,
                    "timestamp": time.strftime("%H:%M:%S")
                }
            else:
                return {"status": "no_targets", "message": "Gökyüzü şu an temiz. Hedef bulunamadı."}
        else:
            return {"status": "error", "message": f"Radar bağlantı hatası: {response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": f"Sensör okuma hatası: {str(e)}"}

if __name__ == "__main__":
    # Test
    print(get_live_radar_data())
