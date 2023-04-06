# import necessary library
import xml.etree.ElementTree as elemTree
import matplotlib.pyplot as plt
import numpy as np

def data_fitting(X,Y,N): # Approximation function definition
    cof=np.polyfit(X,Y,N) # regression (result -> list of coefficient)
    fit_data=np.zeros([X.size])  # define default matrix
    for i in range(N+1): # repeat as degree of polynomial
        fit_data+=(X**(N-i)*cof[i]) # data approximated
    return abs(fit_data) # return absolute value of data
def R_square(X,Y,Y_reg): # obtain R square
    Y_mean=sum(Y)/Y.size # mean of Current
    SST=sum((Y-Y_mean)**2) # total sum of square ( sum of square of difference between data and mean )
    SSE=sum((Y_reg-Y_mean)**2) # Residual sum of square ( sum of square of difference between )
    return SSE/SST # return R sqaure

tree = elemTree.parse("HY202103_D07_(0,0)_LION1_DCM_LMZC.xml") # objectify XML file
root = tree.getroot() # get root from tree

for i in root.iter('Current'): # parsing I, V data
    I=np.array(list(map(float,i.text.split(','))))
    I=abs(I) # absolute value of Current data
for i in root.iter('Voltage'):
    V=np.array(list(map(float,i.text.split(','))))

plt.subplot(1,2,1) # plot on one GUI (regression graph)
data_fitted=data_fitting(V,I,16) # fitted data using defined function
plt.plot(V,data_fitted,'k--',label='best-fit') # plot approximated graph as a dotted line
plt.plot(V,I,'ro',label='data') # plot I-V graph as points
plt.yscale('logit') # set up y axis scale as log

# print('{:.20f}'.format(R_square(V,I,data_fitting(V,I,16))))

# set up background of graph
plt.xlabel('Voltage[V]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.ylabel('Current[A]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.title('IV analysis', fontdict = {'weight': 'bold', 'size':10})
plt.legend(loc='upper left',fontsize=7) # show legend
plt.xticks(fontsize=6) # modulate axis label's fontsize
plt.yticks(fontsize=6)
# show particular data using text method in mathplotlib library
plt.text(0.02,0.8,'R_square = {:.15f}'.format(R_square(V,I,data_fitted)),fontsize=8,transform=plt.gca().transAxes)
# plt.gca().transAxes -> help set up the position of text(x: 0~1, y:0~1)
plt.text(-2,data_fitted[0]*1.5,'{:.11f}'.format(data_fitted[0]),fontsize=6)
plt.text(-1,data_fitted[4]*1.5,'{:.11f}'.format(data_fitted[4]),fontsize=6)
plt.text(1,data_fitted[12]*1.5,'{:.11f}'.format(data_fitted[12]),fontsize=6)

# ---------------------------------------------------------------------------------------------------------------------
plt.subplot(1,2,2) # plot on one GUI (transmission graph)

color=['b','g','r','c','m','y','k'] # variable of color
temp=0 # temporary variable of number of repetition

for i in root.iter('WavelengthSweep'): # data parsing using iterator
    wavelength=np.array(list(map(float,i.find('L').text.split(','))))
    gain=np.array(list(map(float,i.find('IL').text.split(','))))
    bias=i.attrib['DCBias'] # legend of each graph
    if temp==6: # plot reference graph with another way of naming label
        plt.plot(wavelength,gain,label='reference(0V)',color=color[temp])
        continue
    plt.plot(wavelength,gain,label=bias+'V',color=color[temp])
    temp+=1 # to change color, plus 1 to number of repetition variable

# set up the background of graph
plt.xlabel('Wavelength[nm]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.ylabel('Gain[dB]', labelpad=8 , fontdict={'weight': 'bold', 'size':8})
plt.title('Transmission graph', fontdict = {'weight': 'bold', 'size':10})
plt.legend(ncol=4,loc='lower center',fontsize=5) # show legend
plt.xticks(fontsize=6) # modulate axis label's fontsize
plt.yticks(fontsize=6)
plt.show() # show graph to user