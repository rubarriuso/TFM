import os
import time

import sqlite3

from fleet_manager.models.vessels import Vessels
from fleet_manager.models.vessel_events import Vessel_events

class Model:
    """Clase que gestiona como se guardan los datos en la base de datos y las llamadas a las APIs mediante diferentes metodos"""

    def check_vessel(self, imo):
        """Metodo para obtener el id del barco mediante el imo"""
        search_by = f'imo="{imo}"'
        vessel = Vessels(search_by)
        name = vessel.get_name()
        GFWid = vessel.get_id()
        flag = vessel.get_flag()
        return name, GFWid, flag
    

    def create_vessel_entry(self, imo, name, GFWid, flag, time_range=365):
        """Metodo para guardar en la base de datos la información del barco en la tabla VESSELS, 
        así como las posiciones de pesca en la tabla FISHING_ACTIVITY y las visitas de puerto en PORT_VISITS"""        
        try:
            path = os.path.join("fleet_manager", "db","fleet.db")
            connection = sqlite3.connect(path)
            cursor = connection.cursor()
            try:
                #Primero guardamos los datos basicos del barco
                cursor.execute("INSERT INTO VESSELS (IMO, Name, GFWid, Flag) VALUES (?, ?, ?, ?)", (int(imo), name, GFWid, flag))
                connection.commit()
                connection.close()
            except Exception as e:
                connection.close()
                print(e)
                return type(e).__name__

            print(f"Vessel {name} info succesfully saved")

            print(f"Getting fishing positions for {name}")

            self.save_fishing_pos(GFWid, imo, time_range=time_range) #guardamos los datos de las posiciones de pesca
            print(f"Getting port visits for {name}")
            self.save_port_visits(GFWid, imo, time_range=time_range) #guardamos visitas a puerto
            print("Done")
            
        except Exception as e:
            #Si hay un error borramos los datos basicos del barco
            #No quiero guardar el barco si no puedo obtener el resto de la información
            path = os.path.join("fleet_manager", "db","fleet.db")
            connection = sqlite3.connect(path)
            cursor = connection.cursor()
            cursor.execute("DELETE FROM VESSELS WHERE IMO = ?", (imo,))            
            connection.commit()
            connection.close()
            print(e)
            return type(e).__name__

    
    def save_fishing_pos(self, GFWid, imo, time_range=365):
        """Metodo que obtiene los eventos de pesca y los guarda"""
        path = os.path.join("fleet_manager", "db","fleet.db")
        connection = sqlite3.connect(path)
        cursor = connection.cursor()
        events = Vessel_events(GFWid, limit=1000, offset=5, activity="FISHING", time_range=time_range)

        data = events.get_data()
        print(len(data["entries"]), "fishing events found")
        
        if len(data["entries"]) != 0: #Si no tiene eventos de pesca simplemente cerrar la base de datos
            for i in range(len(data["entries"])):

                latitude = data["entries"][i]["position"]["lat"]
                longitude = data["entries"][i]["position"]["lon"]
                date = data["entries"][i]["start"][:-5]
                GFWEventID = data["entries"][i]["id"]
                fao_region = sorted(data["entries"][i]["regions"]["fao"], key=len, reverse=True)[0]

                #Usamos ignore para asi poder usar esta misma funcion para actualizar las posicones si queremos
                cursor.execute("INSERT OR IGNORE INTO FISHING_ACTIVITY (GFWEventID, IMO, Date, Lat, Lon, FAO) VALUES (?, ?, ?, ?, ?, ?)", (GFWEventID ,int(imo), date, latitude, longitude, fao_region))

            connection.commit()

        connection.close()


    def save_port_visits(self, GFWid, imo, time_range=365):
        #Lo mismo que con la pesca pero con los puertos
        path = os.path.join("fleet_manager", "db","fleet.db")
        connection = sqlite3.connect(path)
        cursor = connection.cursor()

        events = Vessel_events(GFWid, limit=1000, offset=5, activity="PORT_VISIT", time_range=time_range)
        data = events.get_data()
        
        print(len(data["entries"]), "port visits found")

        if len(data["entries"]) != 0:
            for i in range(len(data["entries"])):
                data_entry = data["entries"][i]
                latitude = data_entry["position"]["lat"]
                longitude = data_entry["position"]["lon"]
                date = data_entry["end"]
                activity =data_entry["type"]
                port_name = data_entry["port_visit"]["intermediateAnchorage"]["name"]
                visit_duration = data_entry["port_visit"]["durationHrs"]
                GFWEventID = data_entry["id"]
                cursor.execute("INSERT OR IGNORE INTO PORT_VISITS (GFWEventID, IMO, Port_name, Duration, Date ,Lat, Lon) VALUES (?, ?, ?, ?, ?, ?, ?)", (GFWEventID, int(imo), port_name, visit_duration, date, latitude, longitude))

            connection.commit()

        connection.close()
    

    def update_last_positions(self):
        """funcion que usa las funciones anteriores para actualizar las visitas a puerto y las posiciones de pesca
        de todos los barcos que hay en la base de datos, se ha puesto time sleep para no exceder el limite de llamadas de GFW"""
        
        path = os.path.join("fleet_manager", "db", "fleet.db")
        connection = sqlite3.connect(path)
        cursor = connection.cursor()
        cursor.execute("SELECT IMO, Name, GFWid FROM VESSELS")
        results = cursor.fetchall()

        for row in results:
            imo, name, GFWid = row
            print(name)
            print("Getting fishing positions...")
            self.save_fishing_pos(GFWid, imo, time_range=90) #Se busca en los ultimos 3 meses
            print("Getting port visits...")
            self.save_port_visits(GFWid, imo, time_range=90)
            time.sleep(1)

        connection.commit()
        connection.close()

    def delete_all_vessels(self):
        """Borra todos los registros de barcos"""

        path = os.path.join("fleet_manager", "db", "fleet.db")
        connection = sqlite3.connect(path)
        cursor = connection.cursor()

        try:
            cursor.execute("DELETE FROM VESSELS;")
            cursor.execute("DELETE FROM PORT_VISITS;")
            cursor.execute("DELETE FROM FISHING_ACTIVITY;")
            connection.commit()
            print("All vessels deleted successfully from all tables.")
        except Exception as e:
            print(f"Error deleting vessels: {e}")
        finally:
            connection.close()