import requests
import json
from fleet_manager.models.GFW_auth import GFW_auth


class Vessels:
    """Clase que busca un barco pesquero en funcion de su IMO en la API de GFW"""
    def __init__(self, search_by):
        headers = GFW_auth().headers


        url ="https://gateway.api.globalfishingwatch.org/v3/vessels/search"

        params = {
            "datasets[0]":'public-global-vessel-identity:latest',
            "where": search_by
        }

        self.response = requests.get(url, params=params, headers=headers)

        if self.response.status_code == 200:
            self.data = self.response.json()
        else:
            print("Error:", self.response.json())

    def get_id(self):
        id = self.data["entries"][0]["selfReportedInfo"][0]["id"]
        return id
    def get_name(self):
        name = self.data["entries"][0]["selfReportedInfo"][0]["shipname"]
        return name
    def get_flag(self):
        flag = self.data["entries"][0]["selfReportedInfo"][0]["flag"]
        return flag
    def get_imo(self):
        imo = self.data["entries"][0]["selfReportedInfo"][0]["imo"]
        return imo
    
    def get_all_data(self):
        all_data = json.dumps(self.data, indent=4)
        return all_data




