import matplotlib.pyplot as plt
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
import torchvision.transforms as transforms
from PIL import Image
import plotly.express as px
import io
import base64
import numpy as np


def find_gps(hdf5):
    data = TimeSeries.read(hdf5, format="hdf5.gwosc")
    return data.t0.value



def plot_show_strain(hdf5):

    data = TimeSeries.read(hdf5, format="hdf5.gwosc")
    gps = data.t0.value
    data = data.whiten()
    plot = data.plot()
    ax = plot.gca()
    ax.set_epoch(gps)
    plot.show()

#plot_show_strain(hdf5="L-L1_GWOSC_O2_4KHZ_R1-1186738176-4096.hdf5")


# # trial
# start = 3683.5
# end = 3687.5
# gps = 1186738176
# target = 1186741861.5
# tdata = TimeSeries.read("L-L1_GWOSC_O2_4KHZ_R1-1186738176-4096.hdf5", format="hdf5.gwosc", start=gps+start, end=gps+end)
# white = tdata.whiten().bandpass(30, 400)
# whiteq = white.q_transform(frange=(10, 1000), outseg=(target-0.4, target+0.4))
# plot2 = whiteq.plot()
# ax1 = plot2.gca()
# ax1.set_epoch(gps)
# ax1.set_ylim(10, 1000)
# ax1.set_yscale("log")
# plot2.show()







# 1. Plotting the time-domain strain data in duration 4s

def plot_4s_strain(hdf5, start, end):

    data = TimeSeries.read(hdf5, format="hdf5.gwosc")
    gps = data.t0.value
    tdata = TimeSeries.read(hdf5, format="hdf5.gwosc", start=gps+start, end=gps+end)
    plot = tdata.plot(figsize=[8, 4])
    ax = plot.gca()
    ax.set_epoch(gps)
    ax.set_title("Time-domain")
    ax.set_ylabel("Strain Amplitude []")
    buf1 = io.BytesIO()
    plot.save(buf1, format="png")
    plot.close()

    target = (2 * gps + start + end) / 2
    tq = tdata.q_transform(frange=(10, 1000), outseg=(target-2, target+2))
    plot = tq.plot(figsize=[8, 4])
    ax = plot.gca()
    ax.set_title("Q-Transformed")
    ax.set_yscale("log")
    ax.set_ylim(10, 1000)
    ax.set_epoch(gps)
    buf2 = io.BytesIO()
    plot.save(buf2, format="png")
    plot.close()

    img1 = Image.open(buf1)
    img2 = Image.open(buf2)
    concat = Image.new("RGB", (img1.width, img1.height+img2.height))
    concat.paste(img1, (0, 0))
    concat.paste(img2, (0, img1.height))

    # buf = io.BytesIO()
    # plot.save(buf, format="png")
    # plot.close()
    # buf.seek(0)
    buf = io.BytesIO()
    concat.save(buf, format="PNG")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii").replace("\n", "")

    return "data:image/png;base64,{}".format(encoded)




def plot_processed_spectrogram(hdf5, start, end, save=False, to_predict=False):

    tranges = [0.5, 1, 2]
    transform = transforms.Grayscale()

    data = TimeSeries.read(hdf5, format="hdf5.gwosc")
    gps = data.t0.value
    tdata = TimeSeries.read(hdf5, format="hdf5.gwosc", start=gps+start, end=gps+end)

    target = (2*gps+start+end)/2

    tq1 = tdata.q_transform(frange=(10, 1000), outseg=(target-tranges[0], target+tranges[0]))
    plot1 = tq1.plot(figsize=[5, 5])
    #plot.colorbar(label="Normalised energy")
    ax1 = plot1.gca()
    ax1.set_epoch(gps)
    ax1.set_ylim(10, 1000)
    ax1.set_yscale("log")
    ax1.grid(False, axis='y', which='both')
    ax1.axis('off')
    buf = io.BytesIO()
    # figname1 = str(int(target)) + "_r.png"
    plot1.save(buf, format="png", bbox_inches="tight", pad_inches=0)
    plot1.close()
    r = transform(Image.open(buf))

    tq2 = tdata.q_transform(frange=(10, 1000), outseg=(target-tranges[1], target+tranges[1]))
    plot2 = tq2.plot(figsize=[5, 5])
    #plot.colorbar(label="Normalised energy")
    ax2 = plot2.gca()
    ax2.set_epoch(gps)
    ax2.set_ylim(10, 1000)
    ax2.set_yscale("log")
    ax2.grid(False, axis='y', which='both')
    ax2.axis('off')
    buf = io.BytesIO()
    # figname2 = str(int(target)) + "_g.png"
    plot2.save(buf, bbox_inches="tight", pad_inches=0)
    plot2.close()
    g = transform(Image.open(buf))

    tq4 = tdata.q_transform(frange=(10, 1000), outseg=(target-tranges[2], target+tranges[2]))
    plot4 = tq4.plot(figsize=[5, 5])
    #plot.colorbar(label="Normalised energy")
    ax4 = plot4.gca()
    ax4.set_epoch(gps)
    ax4.set_ylim(10, 1000)
    ax4.set_yscale("log")
    ax4.grid(False, axis='y', which='both')
    ax4.axis('off')
    buf = io.BytesIO()
    # figname4 = str(int(target)) + "_b.png"
    plot4.save(buf, bbox_inches="tight", pad_inches=0)
    plot4.close()
    b = transform(Image.open(buf))

    combination = Image.merge('RGB', (r, g, b))

    if save:
        figname = str(int(target)) + "_transformed.png"
        combination.save(figname)

    buf = io.BytesIO()
    combination.save(buf, format="png")

    if to_predict:
        return buf

    else:
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii").replace("\n", "")
        return "data:image/png;base64,{}".format(encoded)







# For confident events
def plot_processed_spectrogram_event(event, save_q=False):

    tranges = [0.5, 1, 2]
    transform = transforms.Grayscale()
    gps = event_gps(event)
    print(event + " GPS: ", gps)

    ldata = TimeSeries.fetch_open_data('L1', int(gps)-4, int(gps)+4, cache=True)
    print(event + " data")
    print(ldata)

    imgs = []

    for i in tranges:
        lq = ldata.q_transform(frange=(10, 1000), outseg=(gps-i, gps+i))
        plot = lq.plot(figsize=[4, 4])
        ax = plot.gca()
        ax.set_epoch(gps)
        ax.set_ylim(10, 1000)
        ax.set_yscale("log")
        if save_q:
            plot.colorbar(label="Normalised energy")
            imgname = event + "_" + str(int(i*2)) + "s.png"
            plot.save(imgname)

        ax.grid(False, axis='y', which='both')
        ax.axis('off')
        buf = io.BytesIO()
        plot.save(buf, format="png", bbox_inches="tight", pad_inches=0)
        imgs.append(buf)

    r = transform(Image.open(imgs[0]))
    g = transform(Image.open(imgs[1]))
    b = transform(Image.open(imgs[2]))

    combination = Image.merge('RGB', (r, g, b))
    figname = event + "_transformed.png"
    combination.save(figname)

    return combination







# gps = event_gps("GW150914")
# print(" GPS: ", gps)
#
# ldata = TimeSeries.fetch_open_data('L1', int(gps) - 4, int(gps) + 4, cache=True)
# print("data")
# print(ldata)
#
# lq = ldata.q_transform(frange=(10, 1000), qrange=(5, 5), outseg=(gps-0.5, gps+0.5))
# plot = lq.plot()
# ax = plot.gca()
# ax.set_epoch(gps)
# ax.set_ylim(10, 1000)
# ax.set_yscale("log")
# plot.show()





# plot_show_strain(hdf5="L-L1_GWOSC_O3a_4KHZ_R1-1238175744-4096.hdf5")
# strain = plot_4s_strain(hdf5="L-L1_GWOSC_O3a_4KHZ_R1-1238175744-4096.hdf5", start=2560, end=2564)
# combi = plot_processed_spectrogram(hdf5="L-L1_GWOSC_O3a_4KHZ_R1-1238175744-4096.hdf5", start=2558, end=2562)
# first_gw = plot_processed_spectrogram_event("GW150914", save_q=True)



# 1.1. First plot the time-domain graph of the strain data

# plot = data.plot()
# ax = plot.gca()
# ax.set_epoch(gps)
# plot.show()



# Once select a range of interested period (4 seconds), perform q transform

# q = data.q_transform(frange=(10, 1000))
# plot = q.plot()
# plot.colorbar(label="Normalised energy")
# ax = plot.gca()
# ax.set_epoch(gps)
# ax.set_ylim(10, 1000)
# ax.set_yscale("log")
# # ax.grid(False, axis='y', which='both')
# # ax.axis('off')
# plot.show()





#
# for i in tranges:
#     lq = data.q_transform(frange=(10, 1000), outseg=(gps-i, gps+i))
#     plot = lq.plot(figsize=[4, 4])
#     #plot.colorbar(label="Normalised energy")
#     ax = plot.gca()
#     ax.set_epoch(gps)
#     ax.set_ylim(10, 1000)
#     ax.set_yscale("log")
#     ax.grid(False, axis='y', which='both')
#     ax.axis('off')
#     plot.show()
#     imgname = "try_" + str(int(i*2)) + "s.png"
#     plot.save(imgname, bbox_inches="tight", pad_inches=0)



# 2. Convert into 1s, 2s, 4s spectrograms

# 2.1 Importing event strain data through TimeSeries.fetch_open_data

# gps = event_gps('GW150914')
# print("GW150914 GPS:", gps)
#
# ldata = TimeSeries.fetch_open_data('L1', int(gps)-4, int(gps)+4, cache=True)
# print("GW150914 data")
# print(ldata)



# for i in tranges:
#     lq = ldata.q_transform(frange=(10, 1000), outseg=(gps-i, gps+i))
#     plot = lq.plot(figsize=[4, 4])
#     #plot.colorbar(label="Normalised energy")
#     ax = plot.gca()
#     ax.set_epoch(gps)
#     ax.set_ylim(10, 1000)
#     ax.set_yscale("log")
#     ax.grid(False, axis='y', which='both')
#     ax.axis('off')
#     plot.show()
#     imgname = "GW150914_" + str(int(i*2)) + "s.png"
#     plot.save(imgname, bbox_inches="tight", pad_inches=0)




# 3. Convert the spectrograms into greyscale

# for i in tranges:
#     img = Image.open("GW150914_" + str(int(i*2)) + "s.png")
#     transform = transforms.Grayscale()
#     img = transform(img)
#     img = img.save("GW150914_" + str(int(i*2)) + "s_greyscale.png")


# 4. Combine them to be a new spectrogram
# 1s as R channel, 2s as G channel, 4s as B channel

# r = Image.open("GW150914_" + str(int(tranges[0]*2)) + "s_greyscale.png")
# g = Image.open("GW150914_" + str(int(tranges[1]*2)) + "s_greyscale.png")
# b = Image.open("GW150914_" + str(int(tranges[2]*2)) + "s_greyscale.png")
#
# combination = Image.merge('RGB', (r, g, b))
# combination.save("GW150914_transformed.png")


# 5. Upload to predict

