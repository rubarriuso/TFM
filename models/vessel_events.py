import requests
from datetime import datetime, timedelta
from fleet_manager.models.GFW_auth import GFW_auth

class Vessel_events:

    """Clase que busca los eventos de un barco pesquero por su identificador de GFW para el barco"""
    def __init__(self, id, time_range=365, limit=1500, offset=5, activity="FISHING"):
        #Time range es para poner cuanto tiempo atrás buscar
        url="https://gateway.api.globalfishingwatch.org/v3/events"

        headers=GFW_auth().headers

        current_date = datetime.now().strftime("%Y-%m-%d")

        past_date = (datetime.now() - timedelta(days=time_range)).strftime("%Y-%m-%d")

        activity_dict={"FISHING":"public-global-fishing-events:latest",
                       "PORT_VISIT": "public-global-port-visits-c2-events:latest",
                       "GAP": "public-global-gaps-events:latest",
                       "ENCOUNTER":"public-global-encounters-events:latest"}
        
        params = {
            "limit":f"{limit}",
            "offset":f"{offset}",
            "sort":"-start",
            "datasets[0]":activity_dict[activity],
            "vessels[0]":id,
            "start-date":past_date,
            "end-date":current_date
        }

        self.response = requests.get(url, params=params, headers=headers)

        if self.response.status_code == 200:
            self.data = self.response.json()
        else:
            print("Error:", self.response.json())
    
    def get_data(self):
        return self.data




