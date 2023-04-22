import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from fitting import *

def plot(x):
    tree = ET.parse(x)
    grid = (9, 9)
    plt.figure(figsize=(8, 8))
    plt.subplot2grid(grid, (0, 0), rowspan=4, colspan=4)

    for i in range(1, 7):
        L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
        IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
        L_i = L.text.split(",")
        IL_i = IL.text.split(",")
        L_list_i = list(map(float, L_i))
        IL_list_i = list(map(float, IL_i))
        DBias = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
        plt.plot(L_list_i, IL_list_i, ".", label=DBias.get("DCBias"))

    L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/L")
    IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep/IL")
    L_7 = L.text.split(",")
    IL_7 = IL.text.split(",")
    L_list_7 = list(map(float, L_7))
    IL_list_7 = list(map(float, IL_7))
    DBias = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator[2]/PortCombo/WavelengthSweep")
    plt.plot(L_list_7, IL_list_7, ".", label="reference")
    plt.legend(loc=(0, 0))
    plt.title("Transmission spectra - as measured")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Measured transmission [dB]')

    plt.subplot2grid(grid, (0, 5), rowspan=4, colspan=4)

    plt.scatter(L_list_7, IL_list_7, s=15, label="reference", alpha=0.01, facecolor='none', edgecolor='r')

    polyfiti = np.polyfit(L_list_7, IL_list_7, 6)
    fiti = np.poly1d(polyfiti)
    x = polyfitT(L_list_7, IL_list_7, 6)
    plt.plot(L_list_7, fiti(L_list_7), label="{}th R^2 = {}".format(6, '%0.5f'% x))
    plt.legend(loc="best")
    plt.title("Reference fitting")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Measured transmission [dB]')

    plt.subplot2grid(grid, (5, 0), rowspan=4, colspan=4)

    polyfit6 = np.polyfit(L_list_7, IL_list_7, 6)
    fit6 = np.poly1d(polyfit6)

    for i in range(1, 7):
        L = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/L".format(i))
        IL = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]/IL".format(i))
        L_i = L.text.split(",")
        IL_i = IL.text.split(",")
        L_list_i = list(map(float, L_i))
        IL_list_i = list(map(float, IL_i))
        DBias = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep[{}]".format(i))
        plt.plot(L_list_i, IL_list_i - fit6(L_list_i), ".", label=DBias.get("DCBias"))

    plt.plot(L_list_7, IL_list_7 - fit6(L_list_7), ".",label = DBias.get("DCBias"))
    plt.legend(loc=(0, 0))
    plt.title("Transmission spectra - as processed")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('flat Measured transmission [dB]')

    plt.subplot2grid(grid, (5, 5), rowspan=4, colspan=4)

    from lmfit import Model

    b = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Voltage")
    c = tree.find(".ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement/Current")
    x_2 = b.text.split(",")
    y_2 = c.text.split(",")
    x_list = list(map(float, x_2))
    y_list = list(map(float, y_2))
    y_list_1 = []
    for i in range(len(y_list)):
        g = abs(y_list[i])
        y_list_1.append(g)
    plt.plot(x_list, y_list_1, "ro", label='initial fit')
    plt.yscale("log")
    plt.title("IV-analysis")
    plt.xlabel('Voltage [V]')
    plt.ylabel('Current [A]')

    gmodel = Model(IVfittting)
    result = gmodel.fit(y_list_1, x=x_list, q=1, w=1, alp=1, xi = x_list, yi = y_list_1)

    yhat = result.best_fit
    ybar = np.sum(y_list_1) / len(y_list_1)
    ssreg = np.sum((yhat - ybar) ** 2)
    sstot = np.sum((y_list_1 - ybar) ** 2)
    results = ssreg / sstot

    plt.plot(x_list, result.best_fit, 'b-', label='best fit R^2={}'.format('%0.5f'% results))
    plt.title('IV-fitting')
    plt.text(-1, result.best_fit[4], str(result.best_fit[4]), color='g', horizontalalignment='center',
             verticalalignment='bottom')
    plt.text(1, result.best_fit[12], str(result.best_fit[12]), color='g', horizontalalignment='center',
             verticalalignment='bottom')
    plt.legend()