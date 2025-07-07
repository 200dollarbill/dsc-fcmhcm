import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Button, Label, messagebox, Entry, Radiobutton

y_min, y_max = 0.001, 1000
xmin, xmax = 0.001, 1000

#points_custom = np.random.uniform(low=[xmin, y_min], high=[xmax, y_max], size=(1000, 2))
#print(points_custom[:5])

#plt.scatter(points_custom[:, 0], points_custom[:, 1], s=1, c='blue', alpha=0.5)
#plt.show()

root = tk.Tk()
root.geometry("300x400")

cmeans = tk.IntVar()
cmeans.set(1)
xval = 70
yval = 10



def graph_click():
    # cmeans_type = cmeans.get()
    global Npoints
    global epoch
    epoch = int(epochInput.get())
    n_count = int(nInput.get())
        
    groupcount = int(groupInput.get())
    #y_max,xmax = nInput.get(),nInput.get()
    Npoints = np.random.uniform(low=[xmin, y_min], high=[xmax, y_max], size=(n_count, 2))
    #messagebox.showinfo("Input n is " + ncount, "CM is " + str(cmeans_type))
    plt.scatter(Npoints[:, 0], Npoints[:, 1], s=1, alpha=0.5)
    plt.title("Setup")
    plt.show()
    

    

def calculate_cmeans():
    n_points = int(nInput.get()) - 1
    k = int(groupInput.get())

    #points_custom = np.random.uniform(low=[xmin, y_min], high=[xmax, y_max], size=(n_points, 2))
    
    clustering_type = cmeans.get()
    #hcm
    #hcm
    if clustering_type == 1:
        index = np.random.choice(n_points, k, replace=False)
        centroid0 = Npoints[index]

        # 1 epoch
        for _ in range(epoch):
            distances = np.sqrt(((centroid0[:, np.newaxis] - Npoints) ** 2).sum(axis=2))
            cluster_group = np.argmin(distances, axis=0)
            
            centroid1 = np.array([Npoints[cluster_group == i].mean(axis=0) if np.sum(cluster_group == i) > 0 else centroid0[i] for i in range(k)])

            #update centroid
            if np.all(centroid0 == centroid1):
                break
            centroid0 = centroid1
        header = "Data Point | Cluster"
        print(header)
        for i, group in enumerate(cluster_group):
            print(f"    {i+1:<8} |        {group + 1}") 

        plt.figure(figsize=(8, 6))
        colors = plt.get_cmap('plasma', k)
        for i in range(k):
            cluster_points = Npoints[cluster_group == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=colors(i), alpha=0.8, label=f'Group {i+1}')
        plt.scatter(centroid0[:, 0], centroid0[:, 1], s=150, c='red', marker='1', label='centroid')
        plt.title("HCM Clustering")
        plt.legend()
        plt.grid(True)
        plt.show()


    #fcm
    elif clustering_type == 2:
        # 
        n_points+= 1
        m = 2.0
        
        U = np.random.rand(n_points, k)
        U = U / np.sum(U, axis=1, keepdims=True)
        
        centroid0 = np.zeros((k, 2))
#        # 1 epoch
        for _ in range(epoch):
            U_m = U ** m
            centroid0 = np.dot(U_m.T, Npoints) / np.sum(U_m, axis=0)[:, np.newaxis]
            #euclidean distance sum
            distances = np.sqrt(((Npoints - centroid0[:, np.newaxis])**2).sum(axis=2)).T #transpos

            membership = distances ** (-2. / (m - 1))
            U = membership / np.sum(membership, axis=1, keepdims=True)

        header = "Data Point |"
        for i in range(k):
            header += f"  Cluster {i+1}  |"
        print(header)

        for i, u_row in enumerate(U):
            row_str = f"    {i+1:<8} |"
            for val in u_row:
                row_str += f"  {val:.4f}   |"
            print(row_str)

        cluster_group = np.argmax(U, axis=1)
        
        plt.figure(figsize=(8, 6))
        colors = plt.get_cmap('plasma', k)
        for i in range(k):
            cluster_points = Npoints[cluster_group == i]
            if cluster_points.shape[0] > 0:
                plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=5, color=colors(i), alpha=0.8, label=f'Group {i+1}')
        plt.scatter(centroid0[:, 0], centroid0[:, 1], s=150, c='red', marker='1', label='centroid')
        plt.title("FCM Clustering")
        plt.legend()
        plt.grid(True)
        plt.show()

epochInput = Entry(
    root,
    width=10
)
graphButton = Button(
    root,
    text="Setup",
    command=graph_click,
)
calculateButton = Button(
    root,
    text="Calculate and Graph",
    command=calculate_cmeans
)
graphLabel = Label(
    root,
    text="Configurations",
    font=("Arial", 12, "bold"),
)
nInput = Entry(
    root,
    width=10,
)
inputLabel = Label(
    root,
    text="Input n",
    font=("Arial", 12, "bold"),
)
HCM = Radiobutton(
    root,
    text="HCM",
    variable=cmeans,
    value=1,
    #command=HCM_tf,
)
FCM = Radiobutton(
    root,
    text="FCM", 
    variable=cmeans,
    value=2,
    #command=FCM_tf
)
cmeansLabel = Label(
    root,
    text="Clustering Method",
    font=("Arial", 12, "bold"),
)
groupLabel = Label(
    root,
    text="Group count",
    font=("Arial", 12, "bold"),
)
groupInput = Entry(
    root,
    width=10,
)
epochLabel = Label(
    root,
    text="Epoch count",
    font=("Arial", 12, "bold"),
)

graphLabel.place(x=xval, y=yval+10)
FCM.place(x=xval, y=yval+50)
HCM.place(x=xval, y=yval+70)
cmeansLabel.place(x=xval, y=yval+30)
inputLabel.place(x=90, y=yval+90)
nInput.place(x=xval, y=yval+120)
groupLabel.place(x=xval, y=yval+150)
groupInput.place(x=xval, y=yval+180)
graphButton.place(x=xval, y=yval+280)
calculateButton.place(x=xval, y=yval+320)
epochLabel.place(x=xval, y=yval+210)
epochInput.place(x=xval, y=yval+250)


calculateButton.config(width=15, height=1, font=("Arial", 12))
nInput.config(width=15, font=("Arial", 12))
graphButton.config(width=15, height=1, font=("Arial", 12))
groupInput.config(width=15, font=("Arial", 12))
epochInput.config(width=15, font=("Arial", 12))

root.mainloop()