import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from astral import moon

class DataBuilder:
    def __init__(self, target_var, sp, todas=False,
        target_bbox=[-61.0 - 0.5, -47.5 - 0.5, -60.0 + 0.5, -44.875 + 0.5],
        enviroment_bbox=[-61.0 - 0.5, -47.5 - 4, -60.0 + 2, -44.875 + 1],
        target_res=0.125,
        split_year=2023, split_year_test=2025,
        start_date="2009-01-01", end_date="2025-12-31",
        window_size=12,
        batch_size=24,
        data_dir="../data/processed"):
        
        self.sp = sp
        self.todas = todas
        self.target_res = target_res
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = target_bbox
        self.enviroment_min_lon, self.enviroment_min_lat, self.enviroment_max_lon, self.enviroment_max_lat = enviroment_bbox
        self.target_var = target_var
        self.split_year = split_year
        self.split_year_test = split_year_test
        self.min_time = pd.to_datetime(start_date)
        self.max_time = pd.to_datetime(end_date)
        
        self.window_size = window_size
        self.batch_size = batch_size
        self.data_dir = data_dir

    def _cropped_target(self, da):
        return da.sel(
            lon=slice(self.min_lon, self.max_lon),
            lat=slice(self.min_lat, self.max_lat),
            time=slice(self.min_time, self.max_time)
        )

    def _cropped_global(self, da):
        return da.sel(
            lon=slice(self.enviroment_min_lon, self.enviroment_max_lon),
            lat=slice(self.enviroment_min_lat, self.enviroment_max_lat),
            time=slice(self.min_time, self.max_time)
        )

    def _load_and_align_data(self):
        # Target & Mask
        fishing_ds = xr.open_dataset(f"{self.data_dir}/targets/cpue_{self.target_res}.nc").sel(FAOspp=self.sp)
        fishing = fishing_ds[self.target_var].fillna(0)

        mask_ds = xr.open_dataset(f"{self.data_dir}/static/area_pesca_{self.target_res}.nc")
        mask = mask_ds["mask"].fillna(0).broadcast_like(fishing)

        # Dynamic Features
        temp = xr.open_dataset(f"{self.data_dir}/dynamic/to_surface.nc").rename({"to":"TO"})["TO"].fillna(0)
        temp_bottom = xr.open_dataset(f"{self.data_dir}/dynamic/temp_bottom.nc").rename({"to": "TOB"})["TOB"].fillna(0)
        chl = xr.open_dataset(f"{self.data_dir}/dynamic/chl.nc")["CHL"].fillna(0)
        mixed = xr.open_dataset(f"{self.data_dir}/dynamic/mixed_layer.nc").rename({"mlotst":"MLOTST"})["MLOTST"].fillna(0)
        
        zo = xr.open_dataset(f"{self.data_dir}/dynamic/zo_surface.nc").rename({"zo":"ZO"})["ZO"].fillna(0)
        so = xr.open_dataset(f"{self.data_dir}/dynamic/so_surface.nc").rename({"so":"SO"})["SO"].fillna(0)
        ugo = xr.open_dataset(f"{self.data_dir}/dynamic/ugo_surface.nc").rename({"ugo":"UGO"})["UGO"].fillna(0)
        vgo = xr.open_dataset(f"{self.data_dir}/dynamic/vgo_surface.nc").rename({"vgo":"VGO"})["VGO"].fillna(0)
        
        pp = xr.open_dataset(f"{self.data_dir}/dynamic/pp.nc")["PP"].fillna(0)
        cdm = xr.open_dataset(f"{self.data_dir}/dynamic/cdm.nc")["CDM"].fillna(0)
        spm = xr.open_dataset(f"{self.data_dir}/dynamic/spm.nc")["SPM"].fillna(0)
        zsd = xr.open_dataset(f"{self.data_dir}/dynamic/zsd.nc")["ZSD"].fillna(0)

        # Static Features (broadcasted)
        depth = xr.open_dataset(f"{self.data_dir}/static/depth.nc").rename({"depth":"PROF"})["PROF"].fillna(0).broadcast_like(temp)

        # Engineered Features
        month = temp["time"].dt.month
        month_sin = np.sin(2 * np.pi * month / 12).broadcast_like(temp)
        month_cos = np.cos(2 * np.pi * month / 12).broadcast_like(temp)
        month_sin.name = "MSEN"
        month_cos.name = "MCOS"
        month = month.broadcast_like(temp)
        
        year = temp["time"].dt.year.broadcast_like(temp)
        year.name = "Año"

        lat = temp["lat"].broadcast_like(temp)
        lat.name = "LAT"
        lon = temp["lon"].broadcast_like(temp)
        lon.name = "LON"

        times = pd.DatetimeIndex(temp.time.values)
        moon_ = np.array([moon.phase(t) for t in times]) 
        moon_phase = xr.DataArray(moon_, coords={"time": temp.time}, dims=["time"], name="FL").fillna(0).broadcast_like(temp)

        # Align
        temp, temp_bottom, chl, mixed, depth, month_sin, month_cos, lat, lon, zo, so, month, year, ugo, vgo, pp, cdm, spm, zsd, moon_phase = xr.align(
            temp, temp_bottom, chl, mixed, depth, month_sin, month_cos, lat, lon, zo, so, month, year, vgo, ugo, pp, cdm, spm, zsd, moon_phase, join="inner"
        )
        fishing, mask = xr.align(fishing, mask, join="inner")

        # Crop target
        self.fishing = self._cropped_target(fishing).transpose("time", "lat", "lon")
        self.mask = self._cropped_target(mask).transpose("time", "lat", "lon")

        # Define which variables to use
        if self.todas:
            vars_ = [temp, 
                # so,  zo, temp_bottom,
                ugo, vgo, mixed,
                cdm, spm, chl,  #pp, zsd,
                month_sin, month_cos,
                # year,
                # month,
                # moon_phase,
                # lat, lon,
                # depth,
                ]
        else:
            vars_ = [temp, ugo, vgo, mixed, month_sin, month_cos]

        # Crop all selected variables
        self.vars_ = [self._cropped_global(v) for v in vars_]
        # Number of channels and the output dimensions
        self.in_channels = len(self.vars_)
        self.out_hw = self.fishing.shape[1:]

    def _preprocess_and_scale(self):
        self.vars_names = [v.name for v in self.vars_]
        X = xr.concat(self.vars_, dim="channel").transpose("time", "channel", "lat", "lon")
        X = X.assign_coords(channel=self.vars_names)
        
        epsilon = 1e-8 

        # 1. Isolate the training period ONLY to calculate mean/std
        train_slice = X.sel(time=slice(None, f"{self.split_year-1}-12-31"))
        train_mean = train_slice.mean(dim=["time", "lat", "lon"], skipna=True)
        train_std  = train_slice.std(dim=["time", "lat", "lon"], skipna=True)

        # 2. Set mean=0 and std=1 for cyclical features
        for feat in ["MSEN", "MCOS"]:
            if feat in train_mean.coords["channel"].values:
                train_mean.loc[dict(channel=feat)] = 0.0
                train_std.loc[dict(channel=feat)] = 1.0 - epsilon

        # 3. Apply the transformation
        self.X_scaled = (X - train_mean) / (train_std + epsilon)

    def _create_windows(self):
        X_data = self.X_scaled.values   # (time, channels, H, W)
        y_data = self.fishing.values    # (time, H, W)
        m_data = self.mask.values       # (time, H, W)
        time_data = self.fishing.time.values 
        
        X_seq, y_seq, m_seq, time_seq = [], [], [], []

        for i in range(len(X_data) - self.window_size):
            X_seq.append(X_data[i : i + self.window_size])
            y_seq.append(y_data[i + self.window_size])  
            m_seq.append(m_data[i + self.window_size])  
            time_seq.append(time_data[i + self.window_size]) 
            
        return (
            torch.tensor(np.stack(X_seq), dtype=torch.float32),
            torch.tensor(np.stack(y_seq), dtype=torch.float32),
            torch.tensor(np.stack(m_seq), dtype=torch.float32),
            pd.to_datetime(time_seq) 
        )

    def get_loaders(self):

        print("Loading and aligning dataset...")
        self._load_and_align_data()
        
        print("Preprocessing and scaling...")
        self._preprocess_and_scale()
        
        print("Creating rolling windows...")
        X_all, y_all, m_all, t_all = self._create_windows()

        # Split into Train/Val/Test
        train_mask = t_all.year < self.split_year
        val_mask = (t_all.year >= self.split_year) & (t_all.year < self.split_year_test)
        test_mask = t_all.year >= self.split_year_test

        X_train, y_train, m_train = X_all[train_mask], y_all[train_mask], m_all[train_mask]
        X_val, y_val, m_val = X_all[val_mask], y_all[val_mask], m_all[val_mask]
        X_test, y_test, m_test = X_all[test_mask], y_all[test_mask], m_all[test_mask]

        print(f"Train shape: {X_train.shape}")  
        print(f"Val shape:   {X_val.shape}")      
        print(f"Test shape:  {X_test.shape}")    

        # Create Loaders
        train_loader = DataLoader(TensorDataset(X_train, y_train, m_train), batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val, m_val), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test, m_test), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader