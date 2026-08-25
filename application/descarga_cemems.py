import copernicusmarine
import json
import xarray as xr
import numpy as np
import xesmf as xe
from dask.diagnostics import ProgressBar

class CMEMS_Downloader:
    def __init__(self, dryrun=False, res=0.125):
        self.dryrun = dryrun
        self.res = res 
        with open("./data/bbox.json", "r") as f:
            bbox = json.load(f)

        min_lon, min_lat, max_lon, max_lat = [
            bbox["min_lon"],
            bbox["min_lat"],
            bbox["max_lon"],
            bbox["max_lat"]
        ]
        self.min_lon = min_lon
        self.min_lat = min_lat
        self.max_lon = max_lon
        self.max_lat = max_lat



        self.start_date = "2009-01-01"
        self.end_date = None  # hasta la fecha mas reciente disponible

    def download_and_process_phy(self):
        self.download_phy()
        self.process_phy()
        
    def download_and_process_bio(self):
        self.download_bio()
        self.process_bio()

    

    def download_phy(self):
        """"Downlaod CMEMS physical data, uses the multy-year archive (MY) for historical data and near real time for last year data (NRT)"""
        print("Downloading CMEMS physical data...")
        copernicusmarine.subset(
            dataset_id="cmems_obs-mob_glo_phy_my_0.125deg_P1M-m",
            variables=["mlotst", "so", "to", "ugo", "vgo", "zo"], #mixed layer depth, temperature, u and v components of geostrophic current, sea level
            minimum_latitude=self.min_lat, minimum_longitude=self.min_lon,
            maximum_latitude=self.max_lat, maximum_longitude=self.max_lon,
            start_datetime=self.start_date, end_datetime=self.end_date,
            file_format="zarr",
            output_filename = "cmems_obs-mob_glo_phy_my_0.125deg_P1M-m.zarr",
            output_directory="../resources/copernicus_marine_service",
            chunk_size_limit=50_000_000,
            overwrite=True,
            dry_run=self.dryrun # 
        )

        copernicusmarine.subset(
            dataset_id="cmems_obs-mob_glo_phy_nrt_0.125deg_P1M-m",
            variables=["mlotst", "so", "to", "ugo", "vgo", "zo"], #mixed layer depth, temperature, u and v components of geostrophic current, sea level
            minimum_latitude=self.min_lat, minimum_longitude=self.min_lon,
            maximum_latitude=self.max_lat, maximum_longitude=self.max_lon,
            start_datetime=None, end_datetime=None,
            file_format="zarr",
            output_filename = "cmems_obs-mob_glo_phy_nrt_0.125deg_P1M-m.zarr",
            output_directory="../resources/copernicus_marine_service",
            chunk_size_limit=50_000_000,
            overwrite=True,
            dry_run=self.dryrun # False to download and True to only simulate
        )
        print("Download successful.")

    def download_bio(self):
        """"Download biogeochemical data, splti by variable"""
        print("Downloading CMEMS biogeochemical data...")
        copernicusmarine.subset(
            dataset_id="cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M",
            variables=["CHL"],
            minimum_latitude=self.min_lat, minimum_longitude=self.min_lon,
            maximum_latitude=self.max_lat, maximum_longitude=self.max_lon,
            start_datetime=self.start_date, end_datetime=self.end_date,
            file_format="zarr",
            output_filename = "cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M.zarr",
            output_directory="../resources/copernicus_marine_service",
            chunk_size_limit=50_000_000,
            overwrite=True,
            dry_run=self.dryrun # False to download and True to only simulate
        )

        copernicusmarine.subset(
            dataset_id="cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M",
            variables=["PP"],
            minimum_latitude=self.min_lat, minimum_longitude=self.min_lon,
            maximum_latitude=self.max_lat, maximum_longitude=self.max_lon,
            start_datetime=self.start_date, end_datetime=self.end_date,
            file_format="zarr",
            output_filename = "cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M.zarr",
            output_directory="../resources/copernicus_marine_service",
            chunk_size_limit=50_000_000,
            overwrite=True,
            dry_run=self.dryrun # False to download and True to only simulate
        )

        copernicusmarine.subset(
            dataset_id="cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M",
            variables=["SPM", "ZSD"],
            minimum_latitude=self.min_lat, minimum_longitude=self.min_lon,
            maximum_latitude=self.max_lat, maximum_longitude=self.max_lon,
            start_datetime=self.start_date, end_datetime=self.end_date,
            file_format="zarr",
            output_filename = "cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M.zarr",
            output_directory="../resources/copernicus_marine_service",
            chunk_size_limit=50_000_000,
            overwrite=True,
            dry_run=self.dryrun # False to download and True to only simulate
        )

        copernicusmarine.subset(
            dataset_id="cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M",
            variables=["CDM"],
            minimum_latitude=self.min_lat, minimum_longitude=self.min_lon,
            maximum_latitude=self.max_lat, maximum_longitude=self.max_lon,
            start_datetime=self.start_date, end_datetime=self.end_date,
            file_format="zarr",
            output_filename = "cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M.zarr",
            output_directory="../resources/copernicus_marine_service",
            chunk_size_limit=50_000_000,
            overwrite=True,
            dry_run=self.dryrun # False to download and True to only simulate
        )
        print("Download successful.")

    def _regrid_xarray(self, dataset):    
        new_lons = np.arange(self.min_lon, self.max_lon + self.res, self.res)
        new_lats = np.arange(self.min_lat, self.max_lat + self.res, self.res)

        ds_tgt = xr.Dataset({
            'lat': (['lat'], new_lats),
            'lon': (['lon'], new_lons)})

        regridder = xe.Regridder(dataset, ds_tgt, 'conservative')
        with ProgressBar():
            ds_regridded = regridder(dataset).compute()

        return ds_regridded
    
    def process_phy(self):
        """"Process CMEMS physical data, concatenates the two datasets and regrids to regular grid"""

        print("Processing CMEMS physical data...")
        cmems_phy_my = xr.open_zarr("../resources/copernicus_marine_service/cmems_obs-mob_glo_phy_my_0.125deg_P1M-m.zarr", consolidated=True)
        cmems_phy_nrt = xr.open_zarr("../resources/copernicus_marine_service/cmems_obs-mob_glo_phy_nrt_0.125deg_P1M-m.zarr", consolidated=True)
        max_time_my = cmems_phy_my.time.max()
        valid_times_for_phy_nrt = cmems_phy_nrt.time[cmems_phy_nrt.time > max_time_my]
        cmems_phy_nrt_filtered = cmems_phy_nrt.sel(time=valid_times_for_phy_nrt)
        if cmems_phy_nrt_filtered.sizes['time'] > 0:
            cmems_combined = xr.concat([cmems_phy_my, cmems_phy_nrt_filtered], dim="time")
        else:
            cmems_combined = cmems_phy_my
        cmems_combined = cmems_combined.sortby('time')
        cmems_phy = cmems_combined

        cmems_phy_regridded = self._regrid_xarray(cmems_phy)

        save_to = "../resources/copernicus_marine_service/regridded/"
        
        cmems_phy_regridded.to_zarr(save_to+"obs-mob_glo_phy_regridded.zarr", consolidated=True, mode="w")
        print("Processing successful.")
        print(f"Processed CMEMS physical data saved to {save_to}")

    def process_bio(self):
        """"Process CMEMS biogeochemical data"""
        print("Processing CMEMS biogeochemical data...")
        cdm = xr.open_zarr("../resources/copernicus_marine_service/cmems_obs-oc_glo_bgc-optics_my_l4-multi-4km_P1M.zarr")
        chl = xr.open_zarr("../resources/copernicus_marine_service/cmems_obs-oc_glo_bgc-plankton_my_l4-multi-4km_P1M.zarr")
        pp = xr.open_zarr("../resources/copernicus_marine_service/cmems_obs-oc_glo_bgc-pp_my_l4-multi-4km_P1M.zarr")
        transp = xr.open_zarr("../resources/copernicus_marine_service/cmems_obs-oc_glo_bgc-transp_my_l4-multi-4km_P1M.zarr")

        cdm_regridded = self._regrid_xarray(cdm)
        chl_regridded = self._regrid_xarray(chl)
        pp_regridded = self._regrid_xarray(pp)
        transp_regridded = self._regrid_xarray(transp)

        save_to = "../resources/copernicus_marine_service/regridded/"

        cdm_regridded.to_zarr(save_to + "obs-oc_glo_optics.zarr", mode="w")
        chl_regridded.to_zarr(save_to + "obs-oc_glo_plankton.zarr", mode="w")
        pp_regridded.to_zarr(save_to + "regridded/obs-oc_glo_pp.zarr", mode="w")
        transp_regridded.to_zarr(save_to + "obs-oc_glo_transp.zarr", mode="w")
        print("Processing successful.")
        print(f"Processed CMEMS biogeochemical data saved to {save_to}")
        
    