import xarray as xr
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from astral import moon
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display

class DataBuilder:
    def __init__(self, target_var, sp, features,
        target_bbox=[-61.0 - 0.5, -47.5 - 0.5, -60.0 + 0.5, -44.875 + 0.5],
        enviroment_bbox=[-61.0 - 0.5, -47.5 - 4, -60.0 + 2, -44.875 + 1],
        target_res=0.125,
        split_year=2023, split_year_test=2025,
        start_date="2011-01-01", end_date="2025-12-31",
        window_size=12,
        batch_size=24,
        prediction_horizon=1,
        forecast_lead=1,
        anomaly = False,
        data_dir="../data/processed"):
        
        self.sp = sp
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
        self.features = features
        self.prediction_horizon = prediction_horizon
        self.forecast_lead = forecast_lead
        self.history_months = self.window_size + self.forecast_lead - 1
        self.history_min_time = self.min_time - pd.DateOffset(months=self.history_months)
        self.anomaly = anomaly

    def _cropped_target(self, da):
        return da.sel(lon=slice(self.min_lon, self.max_lon), 
                      lat=slice(self.min_lat, self.max_lat), 
                      time=slice(self.history_min_time, self.max_time))

    def _cropped_global(self, da):
        return da.sel(lon=slice(self.enviroment_min_lon, self.enviroment_max_lon), 
                      lat=slice(self.enviroment_min_lat, self.enviroment_max_lat), 
                      time=slice(self.history_min_time, self.max_time))

    def _load_and_align_data(self):
        # Target & Mask
        fishing_ds = xr.open_dataset(f"{self.data_dir}/targets/cpue_{self.target_res}.nc").sel(FAOspp=self.sp)
        fishing = fishing_ds[self.target_var].fillna(0)

        mask_ds = xr.open_dataset(f"{self.data_dir}/static/area_pesca_{self.target_res}.nc")
        mask = mask_ds["mask"].fillna(0).broadcast_like(fishing)

        # Dynamic Features
        temp = xr.open_dataset(f"{self.data_dir}/dynamic/to_surface.nc").rename({"to":"TO"})["TO"].fillna(0)
        temp_bottom = xr.open_dataset(f"{self.data_dir}/dynamic/temp_bottom.nc").rename({"to":"TOB"})["TOB"].fillna(0)
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

        # Static Features
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
        moon_phase = xr.DataArray(moon_, coords={"time":temp.time}, dims=["time"], name="FL").fillna(0).broadcast_like(temp)

        available_vars = {"TO":temp, "TOB":temp_bottom, 
                          "CHL":chl, "MLOTST":mixed, 
                          "ZO":zo, "SO":so, 
                          "UGO":ugo, "VGO":vgo, 
                          "PP":pp, "CDM":cdm, 
                          "SPM":spm, "ZSD":zsd, 
                          "PROF":depth, "MSEN":month_sin, 
                          "MCOS":month_cos, "month":month, 
                          "Año":year, "LAT":lat, 
                          "LON":lon, "FL":moon_phase}

        try:
            selected_raw_vars = [available_vars[feat] for feat in self.features]
        except KeyError as e:
            raise ValueError(f"Feature {e} is not available. Choose from: {list(available_vars.keys())}")

        aligned_vars = list(xr.align(*selected_raw_vars, join="inner"))
        fishing, mask = xr.align(fishing, mask, join="inner")

        # Original target
        self.fishing_original = self._cropped_target(fishing).transpose("time", "lat", "lon")
        self.mask = self._cropped_target(mask).transpose("time", "lat", "lon")

        # Climatology calculated EXCLUSIVELY from training targets
        climatology_train = self.fishing_original.sel(time=slice(self.min_time, f"{self.split_year-1}-12-31"))

        if climatology_train.sizes["time"] == 0:
            raise ValueError("No training data available to calculate climatology.")

        self.climatology = climatology_train.groupby("time.month").mean(dim="time", skipna=True)

        # Target used by the model
        if self.anomaly:
            self.fishing = (self.fishing_original.groupby("time.month") - self.climatology).transpose("time", "lat", "lon")
        else:
            self.fishing = self.fishing_original

        self.vars_ = [self._cropped_global(v) for v in aligned_vars]
        self.in_channels = len(self.vars_)
        self.out_hw = self.fishing.shape[1:]
        self.vars_names = [v.name for v in self.vars_]
        
    def _preprocess_and_scale(self):
        X = xr.concat(self.vars_, dim="channel").transpose("time", "channel", "lat", "lon")
        X = X.assign_coords(channel=self.vars_names)
        epsilon = 1e-8
        train_slice = X.sel(time=slice(self.min_time, f"{self.split_year-1}-12-31"))
        train_mean = train_slice.mean(dim=["time", "lat", "lon"], skipna=True)
        train_std = train_slice.std(dim=["time", "lat", "lon"], skipna=True)

        for feat in ["MSEN", "MCOS"]:
            if feat in train_mean.coords["channel"].values:
                train_mean.loc[dict(channel=feat)] = 0.0
                train_std.loc[dict(channel=feat)] = 1.0 - epsilon

        self.X_scaled = (X - train_mean) / (train_std + epsilon)

    def _create_windows(self):
        X_data = self.X_scaled.values
        y_data = self.fishing.values
        y_original_data = self.fishing_original.values
        m_data = self.mask.values
        time_data = pd.DatetimeIndex(self.fishing.time.values)

        X_seq, y_seq, y_original_seq, clim_seq, m_seq, time_seq = [], [], [], [], [], []
        lead_offset = self.forecast_lead - 1
        first_target_start = np.where(time_data >= self.min_time)[0][0]
        last_target_start = len(time_data) - self.prediction_horizon

        for target_start in range(first_target_start, last_target_start + 1):
            input_end = target_start - lead_offset
            input_start = input_end - self.window_size
            target_end = target_start + self.prediction_horizon

            if input_start < 0:
                continue

            target_times = time_data[target_start:target_end]
            climatology_window = self.climatology.sel(month=target_times.month).values

            X_seq.append(X_data[input_start:input_end])
            y_seq.append(y_data[target_start:target_end])
            y_original_seq.append(y_original_data[target_start:target_end])
            clim_seq.append(climatology_window)
            m_seq.append(m_data[target_start:target_end])
            time_seq.append(target_times.values)

        X_seq = torch.tensor(np.stack(X_seq), dtype=torch.float32)
        y_seq = torch.tensor(np.stack(y_seq), dtype=torch.float32)
        y_original_seq = torch.tensor(np.stack(y_original_seq), dtype=torch.float32)
        clim_seq = torch.tensor(np.stack(clim_seq), dtype=torch.float32)
        m_seq = torch.tensor(np.stack(m_seq), dtype=torch.float32)
        time_seq = np.stack(time_seq)

        return X_seq, y_seq, y_original_seq, clim_seq, m_seq, time_seq
    
    def _create_test_windows(self):
        X_data = self.X_scaled.values
        y_data = self.fishing.values
        y_original_data = self.fishing_original.values
        m_data = self.mask.values
        time_data = pd.DatetimeIndex(self.fishing.time.values)

        test_start = pd.Timestamp(f"{self.split_year_test}-01-01")
        test_end = self.max_time
        test_start_idx = np.where(time_data >= test_start)[0][0]
        test_end_idx = np.where(time_data <= test_end)[0][-1]

        first_target_start = test_start_idx - (self.prediction_horizon - 1)
        last_target_start = test_end_idx

        X_seq, y_seq, y_original_seq, clim_seq, m_seq, time_seq = [], [], [], [], [], []

        for target_start in range(first_target_start, last_target_start + 1):
            lead_offset = self.forecast_lead - 1
            input_end = target_start - lead_offset
            input_start = input_end - self.window_size

            if input_start < 0:
                continue

            X_seq.append(X_data[input_start:input_end])

            y_window = np.zeros((self.prediction_horizon, *y_data.shape[1:]), dtype=y_data.dtype)
            y_original_window = np.zeros((self.prediction_horizon, *y_original_data.shape[1:]), dtype=y_original_data.dtype)
            climatology_window = np.zeros((self.prediction_horizon, *y_original_data.shape[1:]), dtype=y_original_data.dtype)
            m_window = np.zeros((self.prediction_horizon, *m_data.shape[1:]), dtype=m_data.dtype)
            t_window = np.full(self.prediction_horizon, np.datetime64("NaT"), dtype="datetime64[ns]")

            for h in range(self.prediction_horizon):
                idx = target_start + h

                if idx >= len(y_data):
                    continue

                target_date = time_data[idx]
                y_window[h] = y_data[idx]
                y_original_window[h] = y_original_data[idx]
                climatology_window[h] = self.climatology.sel(month=target_date.month).values
                t_window[h] = target_date

                if test_start <= target_date <= test_end:
                    m_window[h] = m_data[idx]

            y_seq.append(y_window)
            y_original_seq.append(y_original_window)
            clim_seq.append(climatology_window)
            m_seq.append(m_window)
            time_seq.append(t_window)

        return torch.tensor(np.stack(X_seq), dtype=torch.float32), torch.tensor(np.stack(y_seq), dtype=torch.float32), torch.tensor(np.stack(y_original_seq), dtype=torch.float32), torch.tensor(np.stack(clim_seq), dtype=torch.float32), torch.tensor(np.stack(m_seq), dtype=torch.float32), np.stack(time_seq)
        
    def get_splits(self):
        print("Loading and aligning dataset...")
        self._load_and_align_data()

        print("Preprocessing and scaling...")
        self._preprocess_and_scale()

        print("Creating rolling windows...")
        X_all, y_all, y_original_all, clim_all, m_all, t_all = self._create_windows()

        years = pd.DatetimeIndex(t_all.reshape(-1)).year.to_numpy().reshape(t_all.shape)
        train_mask = np.all(years < self.split_year, axis=1)
        val_mask = np.all((years >= self.split_year) & (years < self.split_year_test), axis=1)

        X_train, y_train, m_train = X_all[train_mask], y_all[train_mask], m_all[train_mask]
        X_val, y_val, m_val = X_all[val_mask], y_all[val_mask], m_all[val_mask]

        self.y_train_original = y_original_all[train_mask]
        self.y_val_original = y_original_all[val_mask]
        self.clim_train = clim_all[train_mask]
        self.clim_val = clim_all[val_mask]
        self.t_train = t_all[train_mask]
        self.t_val = t_all[val_mask]
        self.m_train = m_train
        self.m_val = m_val

        X_test, y_test, y_test_original, clim_test, m_test, t_test = self._create_test_windows()

        self.y_test_original = y_test_original
        self.clim_test = clim_test
        self.t_test = t_test
        self.m_test = m_test

        print(f"Train: X {X_train.shape} | y {y_train.shape}")
        print(f"Val:   X {X_val.shape} | y {y_val.shape}")
        print(f"Test:  X {X_test.shape} | y {y_test.shape}")

        return (X_train, y_train, m_train), (X_val, y_val, m_val), (X_test, y_test, m_test)

    def get_loaders(self):
        # Fetch the raw splits
        (X_train, y_train, m_train), (X_val, y_val, m_val), (X_test, y_test, m_test) = self.get_splits()

        # Create Loaders
        train_loader = DataLoader(TensorDataset(X_train, y_train, m_train), batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(TensorDataset(X_val, y_val, m_val), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test, m_test), batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    def plot_correlations(self):
        """
        Calculates Variance Inflation Factor (VIF), prints suggested variables to drop,
        and plots a correlation matrix heatmap.
        """
        # Ensure data is loaded and scaled before calculating correlations
        if not hasattr(self, 'X_scaled'):
            print("Data not initialized yet. Loading and scaling now...")
            self._load_and_align_data()
            self._preprocess_and_scale()

        # Reshape for correlation and VIF
        X_np = self.X_scaled.values  # (time, channel, lat, lon)
        t, c, h, w = X_np.shape
        X_flat = X_np.reshape(t, c, -1)       # (time, channel, pixels)
        X_flat = X_flat.transpose(0, 2, 1)    # (time, pixels, channel)
        X_flat = X_flat.reshape(-1, c)        # (samples, channel)
        
        df = pd.DataFrame(X_flat, columns=self.vars_names)
        
        # -------- Variance Inflation Factor (VIF) --------
        print("Calculating Variance Inflation Factor (VIF)...")
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        
        print("\n--- VIF Results ---")
        display(vif_data.sort_values("VIF", ascending=False))

        # -------- Correlation Matrix --------
        corr_matrix = df.corr()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column].abs() > 0.75)]
        print(f"\nSuggested variables to drop (Correlation > 0.75): {to_drop}")
        
        # Plotting
        plt.figure(figsize=(8, 7))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, 
                        linewidths=0.5, annot_kws={"size": 8})

        plt.title("Matriz de correlación de las variables", fontsize=14)
        plt.tight_layout()
        plt.show()

    def get_original_targets(self, split):
        data = {"train":self.y_train_original, "val":self.y_val_original, "validation":self.y_val_original, "test":self.y_test_original}
        if split not in data:
            raise ValueError("split must be 'train', 'val', 'validation' or 'test'")
        return data[split]

    def get_climatology_targets(self, split):
        data = {"train":self.clim_train, "val":self.clim_val, "validation":self.clim_val, "test":self.clim_test}
        if split not in data:
            raise ValueError("split must be 'train', 'val', 'validation' or 'test'")
        return data[split]

    def get_split_mask(self, split):
        data = {"train":self.m_train, "val":self.m_val, "validation":self.m_val, "test":self.m_test}
        if split not in data:
            raise ValueError("split must be 'train', 'val', 'validation' or 'test'")
        return data[split]

    def get_split_times(self, split):
        data = {"train":self.t_train, "val":self.t_val, "validation":self.t_val, "test":self.t_test}
        if split not in data:
            raise ValueError("split must be 'train', 'val', 'validation' or 'test'")
        return data[split]

    def reconstruct_predictions(self, predictions, split):
        if not self.anomaly:
            return predictions
        climatology = self.get_climatology_targets(split)
        if isinstance(predictions, np.ndarray):
            return predictions + climatology.cpu().numpy()
        return predictions + climatology.to(predictions.device)