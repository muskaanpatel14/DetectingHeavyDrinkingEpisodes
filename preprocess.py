import os
import pickle
import pandas as pd
import numpy as np
import datetime
# import scipy
# from scipy.stats import skew, kurtosis
import tensorflow as tf

TAC_DATA_FILE_SUFFIX = '_clean_TAC.csv'
TAC_DRUNK_THRESHOLD = 0.08

# Util - gets milisecond value of time
def get_time_value(x):
  # x is ms. it is divided by 1000 to get microsecond
  t = datetime.datetime.fromtimestamp(x/1000.0)
  t = t.replace(microsecond = 0)
  return int(t.timestamp())

# Returns accelerometer data with timestamps as milliseconds
def get_accelerometer_data(pids = [], path = '../data/all_accelerometer_data_pids_13.csv'):
  print("Reading Accelerometer Data:")
  # Read Accelerometer Data
  acc_data = pd.read_csv('../data/all_accelerometer_data_pids_13.csv')

  assert acc_data['time'].is_monotonic_increasing, "Accelerometer Data is not sorted by time"

  acc_data['window10'] = acc_data['time'].apply(get_time_value)
  acc_data = acc_data.drop(columns="time")
  acc_data = acc_data.rename(columns = {"window10": "time"})

  # acc_data.head()
  # pids = acc_data['pid'].unique()

  # print("PID: " + str(pid))
  acc_data = acc_data[acc_data["pid"].isin(pids)]

  print("Accelerometer Data Shape: " + str(acc_data.shape))
  print("Accelerometer Data PIDs: " + ",".join(acc_data['pid'].unique()))
  
  # acc_data_pid.describe() 
  print("-------------------")
  return acc_data

# Get TAC data for a given PID
# Binarizes TAC on a thresold
def get_tac_data(pids = [], path_prefix = '../data/clean_tac/'):
  print("Reading TAC Data:")

  clean_tac_data = pd.DataFrame()
  for pid in pids:
    temp_df = pd.read_csv(path_prefix + pid + TAC_DATA_FILE_SUFFIX)    

    assert temp_df['timestamp'].is_monotonic_increasing, "TAC Data is not sorted by time"

  # Binarizing TAC value to drunk or not drunk based on threshold
    temp_df["tac"] = np.where(temp_df["TAC_Reading"] > TAC_DRUNK_THRESHOLD, 1, 0)

    temp_df = temp_df.drop(columns="TAC_Reading")
    temp_df = temp_df.rename(columns={"tac": "TAC_Reading"})
    temp_df["pid"] = pid
    clean_tac_data = pd.concat([clean_tac_data, temp_df])
  
  print("-------------------")
  return clean_tac_data

# # Up sampling tac data to match acc data
# def upsample_tac_data(tac_data, acc_data):
  
#   tac_ts = tac_data['timestamp'] 
#   acc_ts = acc_data['time']
#   all_labels = list()
#   offset_tac, offset_acc = 0, 0
#   while offset_tac < len(tac_ts) and offset_acc < len(acc_ts):
    
#     while acc_ts.iloc[offset_acc] < tac_ts.iloc[offset_tac]:
#       all_labels.append([tac_data.iloc[offset_tac]['TAC_Reading'], acc_ts.iloc[offset_acc]])
#       offset_acc += 1
#       if offset_acc >= len(acc_ts):
#         break

#     offset_tac += 1

#   upsampled_tac = pd.DataFrame(all_labels, columns = ["tac", "time"])
#   return upsampled_tac

def upsample_and_join_tac_with_acc(tac_data, acc_data, pids):
  final_data = pd.DataFrame()
  for pid in pids:
    print("[Upsampling] " + str(pid))
    tac_data_mod = tac_data[tac_data["pid"] == pid]
    assert tac_data_mod['timestamp'].is_monotonic_increasing, "[Upsampling] TAC Data is not sorted by time for " + str(pid)
    acc_data_mod = acc_data[acc_data["pid"] == pid]
    assert acc_data_mod['time'].is_monotonic_increasing, "[Upsampling] TAC Data is not sorted by time for " + str(pid)
    tac_data_mod = tac_data_mod.drop(columns = "pid")

    tac_data_mod["from"] = tac_data_mod["timestamp"].shift(1, fill_value=-1) + 1
    tac_data_mod.index = pd.IntervalIndex.from_arrays(tac_data_mod["from"], tac_data_mod["timestamp"], closed = "both")
    acc_data_mod['tac'] = acc_data_mod["time"].apply(lambda x: tac_data_mod.iloc[tac_data_mod.index.get_loc(x)]["TAC_Reading"])
    final_data = pd.concat([final_data, acc_data_mod])

  return final_data

def sample_n_values_per_unit_time(data, n = 20):
  t = data.groupby([ "pid", "time"])["time"].agg(["count"])
  og_row_count = len(data)

  print("Max count is " + str(t["count"].max()))
  print("Min count is " + str(t["count"].min()))

  # Filtering
  pid_time_pair = t[t["count"] > n].index.values
  tuples = pd.MultiIndex.from_frame(data[["pid", "time"]])
  data_temp = data[tuples.isin(pid_time_pair)]

  filtered_row_count = len(data_temp)
  print("Dropped " + str(og_row_count - filtered_row_count) + " out of " + str(og_row_count) + " rows while sampling to satisy even sampling rate")
  
  return data_temp.groupby([ "pid", "time"]).sample(n = n)

def create_sliding_window(data,pids, window_size = 10, sample_n = 20):
  data_copy = data.copy()
  final = []
  labels = []
  for pid in pids:
    print("[Sliding Window] " + str(pid))
    temptemp = data_copy[data_copy['pid'] == pid]
    assert temptemp["time"].is_monotonic_increasing, "[Sliding Window] Data is not sorted by time for " + str(pid)
    times = temptemp.time.unique()
    
    final_temp =[]
    labels_temp = []
    for i in range(len(times)):
      time_to_filter = [times[j] if j >= 0 else -1 for j in range(i-(window_size - 1), i+1)]

      temptemptemp = temptemp[temptemp['time'].isin(time_to_filter)]
      x_dash = np.array(temptemptemp["x"])
      y_dash = np.array(temptemptemp["y"])
      z_dash = np.array(temptemptemp["z"])

      x_dash = np.pad(x_dash, ((sample_n * window_size) - len(x_dash), 0), "constant")
      y_dash = np.pad(y_dash, ((sample_n * window_size) - len(y_dash), 0), "constant")
      z_dash = np.pad(z_dash, ((sample_n * window_size) - len(z_dash), 0), "constant")

      # a = np.vstack((temptemptemp["x"].apply(lambda x: np.array(x, dtype="float32")), temptemptemp["y"].apply(lambda x: np.array(x, dtype="float32")), temptemptemp["z"].apply(lambda x: np.array(x, dtype="float32"))))
      a = np.transpose(np.vstack((x_dash, y_dash, z_dash)))
      final_temp.append(a)
      labels_temp.append(temptemptemp.head(1)["tac"])

    final.append(np.array(final_temp))
    labels.append(np.array(labels_temp))
    # final = np.append(final, np.array(final_temp))
    # labels = np.append(labels, np.array(labels_temp))
  
  final_arr = np.asarray(final).astype('float32')
  labels_arr = np.asarray(labels).astype('float32')

  return final_arr, labels_arr


def pd_to_np(data):
  data_copy = data.copy()
  pids = data["pid"].unique()
  final = []
  labels = []
  for pid in pids:
    temptemp = data_copy[data_copy['pid'] == pid]
    times = temptemp.time.unique()
    final_temp =[]
    labels_temp = []
    for i in range(len(times)):
      # time_to_filter = [times[j] if j >= 0 else -1 for j in range(i-(window_size - 1), i+1)]

      temptemptemp = temptemp[temptemp['time'] == times[i]]
      x_dash = np.array(temptemptemp["x"])
      y_dash = np.array(temptemptemp["y"])
      z_dash = np.array(temptemptemp["z"])

      a = np.transpose(np.vstack((x_dash, y_dash, z_dash)))
      final_temp.append(a)
      labels_temp.append(temptemptemp.head(1)["tac"])

    final.append(np.array(final_temp))
    labels.append(np.array(labels_temp))
  
  final_arr = np.asarray(final).astype('float32')
  labels_arr = np.asarray(labels).astype('float32')

  return final_arr, labels_arr

def shuffle(X, Y):
  print("Shuffling data")
  indices = range(len(X))
  indices = tf.random.shuffle(indices)

  X = tf.gather(X, indices)
  Y = tf.gather(Y, indices)

  print("[Done] Shuffling data")
  return X, Y


def create_pickle(X, Y, filename, folder_path = "../data/pickles"):
  d = dict(X = X, Y = Y)
  if os.path.exists(folder_path) ==  False:
    os.makedirs(folder_path)

  with open(folder_path + "/" + filename + ".p", 'wb') as pickle_file:
      pickle.dump(d, pickle_file)
  print("Data has been dumped into " + folder_path + "/" + filename + ".p")

def create_pickle_df(data, filename, folder_path = "../data/pickles"):
  d = dict(data = data)
  if os.path.exists(folder_path) ==  False:
    os.makedirs(folder_path)

  with open(folder_path + "/" + filename + ".p", 'wb') as pickle_file:
      pickle.dump(d, pickle_file)
  print("Data has been dumped into " + folder_path + "/" + filename + ".p")


# MAIN
if __name__ == "__main__":
  # Constants - Sampling
  sampling_rate = 20
  
# For one PID
#     range of sampling frequency raw
#     

# Processed data
# Numbers of records per pid
# 

  # Sampling - Sliding Window
  sliding_window = False
  window_size = 10
  
  # CC6740, DK3500, HV0618, PC6771, MJ8002, SA0297- ignore
  pids = ["BK7610", "BU4707", "DC6359", "JB3156", "JR8022", "MC7070", "SF3079"]
  # pids = ["BK7610", "BU4707", "DC6359"]
  # pids = ["DC6359"]
  filename = "all_pids_data_slidingwindow" + str(sliding_window) + str(window_size) + "_samplingrate" + str(sampling_rate)

  write_pd = True
  if write_pd:
    filename += "_df"

  # Read Data
  acc_data = get_accelerometer_data(pids = pids)
  tac_data = get_tac_data(pids = pids)
  
  print("Upsampling and joining TAC data with Acc Data")
  acc_tac_data = upsample_and_join_tac_with_acc(tac_data, acc_data, pids)
  print("[Done] Upsampling and joining TAC data with Acc Data")

  print("Sampling " + str(sampling_rate) + " records per unit time")
  acc_tac_data = sample_n_values_per_unit_time(acc_tac_data, sampling_rate)
  print("[Done] Sampling " + str(sampling_rate) + " records per unit time")

  print(acc_tac_data.groupby([ "pid"])["time"].agg(["count"]))

  if sliding_window:
    print("Creating sliding window of size " + str(window_size))
    X, Y = create_sliding_window(acc_tac_data,pids, window_size, sampling_rate)
    print("[Done] Creating sliding window of size " + str(window_size))
    X = np.reshape(X, (X.shape[1], X.shape[2], X.shape[3])) 
    Y = np.reshape(Y, (Y.shape[1], Y.shape[2]))

    X, Y = shuffle(X, Y)

    print("Writing data")
    create_pickle(X, Y, filename)
    print("Data Written")
  else:
    if write_pd:
      create_pickle_df(acc_tac_data, filename)
      
    else:
      X, Y = pd_to_np(acc_tac_data)


      X, Y = shuffle(X, Y)

      print("Writing data")
      create_pickle(X, Y, filename)
      print("Data Written")