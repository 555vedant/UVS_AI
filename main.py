import tkinter as tk
from gui import UVSGui
from reposition import pre_train_and_save
from data import Station, Task, Vehicle
# imporve this logic in repo.py
def run_pre_training():
    print("Starting pre-training...")
    stations = [
        Station("M", (0, 0), 5),
        Station("B", (2, 2), 3),
        Station("Q", (4, 4), 4)
    ]
    tasks = [
        Task(*"M-B-6-1-10".split("-")),
        Task(*"B-Q-3-2-15".split("-")),
        Task(*"M-Q-4-1-12".split("-")),
        Task(*"B-M-5-1-15".split("-"))
    ]
    vehicles = [
        Vehicle(*"V1-M-85-5".split("-")),
        Vehicle(*"V2-B-60-3".split("-")),
        Vehicle(*"V3-M-70-4".split("-")),
        Vehicle(*"V4-Q-50-2".split("-"))
    ]
    pre_train_and_save(stations, vehicles, tasks, algo='rdr', episodes=1000)
    pre_train_and_save(stations, vehicles, tasks, algo='rar', episodes=1000)
    print("Pre-training complete. Models saved: rdr_model.h5, rar_actor.h5, rar_critic.h5")

# Run training automatically
# run_pre_training()

if __name__ == "__main__":
    root = tk.Tk()
    app = UVSGui(root)
    root.mainloop()