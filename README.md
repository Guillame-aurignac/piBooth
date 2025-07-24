# Project Brief
Raspberry pi 5 based photo booth.

In addition to taking pictures, filters are applied and the final result is uploaded to an [Immich](https://github.com/immich-app/immich) server.
Then added to custom album (if the id is provided) and custom location (if lat long is provided)

# Hardware
- Raspberry pi 5 (for gpio compatibility else any pi or pi like)
- A switch
- CSI raspberry camera (global shuter model used for testing)
- Display
- Pi power supply

# Prerequisites
After cloning the repo, create a `.env` file at the root and add the following variables

```txt
API_KEY = "your immich API key"
BASE_URL = "https://your immich server url/api"
ALBUM_ID = "immich album id"
latitude = <your lat>
longitude = <your long>
```

# Running
```bash
pip3 install -r requirements.txt
python3 main.py
```