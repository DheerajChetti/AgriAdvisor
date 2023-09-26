

#make some changes in ferttilizer dataset

fert['Crop'] = fert['Crop'].replace('mungbeans','mungbean')
fert['Crop'] = fert['Crop'].replace('lentils(masoordal)','lentil')
fert['Crop'] = fert['Crop'].replace('pigeonpeas(toordal)','pigeonpeas')
fert['Crop'] = fert['Crop'].replace('mothbean(matki)','mothbeans')
fert['Crop'] = fert['Crop'].replace('chickpeas(channa)','chickpea')
crop.head()
crop.tail()
crop_names = crop['label'].unique()
crop_names
fert.head()
del fert['Unnamed: 0']
crop_names_from_fert = fert['Crop'].unique()
crop_names_from_fert


