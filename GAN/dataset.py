import torch
from torch.utils.data import Dataset
from pymatgen.core.composition import Composition


class CompositionDataset(Dataset):
    def __init__(self, data, elements_list, condi, c_value) -> None:
        super().__init__()
        # raw_data = raw_data.iloc[:int(raw_data.shape[0]/20), :]
        self.condi = condi
        data['composition'] = data['composition'].apply(lambda comp: Composition(comp).fractional_composition)
        self.features = self.get_features(data.loc[:, ['composition']], elements_list)
        if condi:
            self.lables = self.get_lablels(data, c_value)

    def get_lablels(self, data, c_value):
        condition = (torch.tensor(data.iloc[:, 1].values) > c_value).unsqueeze(1)
        result = torch.where(condition, torch.tensor([0, 1]), torch.tensor([1, 0]))
        return result

    def get_features(self, data, elements_list):
        for element in elements_list:
            def get_atomic_fraction(comps):
                return comps.get_atomic_fraction(element)
            data[element] = data['composition'].apply(get_atomic_fraction)
        
        return torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)
    
    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        x = self.features[index]
        if self.condi:
            y = self.lables[index]
            return x, y
        return x
        
class CCompositionDataset(Dataset):
    def __init__(self, data, elements_list) -> None:
        super().__init__()
        # raw_data = raw_data.iloc[:int(raw_data.shape[0]/20), :]
        
        data['composition'] = data['composition'].apply(lambda comp: Composition(comp).fractional_composition)
        self.features = self.get_features(data.iloc[:, 0:1], elements_list)
        self.conditions = self.get_conditions(data.iloc[:, 1:2])
        self.labels = torch.tensor(data['E'].values, dtype=torch.float32)
    
    def get_conditions(self, data):
        data['c1'] = data['c2'] = 0
        data.loc[data.loc[: ,'E']<-29, 'c1'] = 1
        data.loc[data.loc[: ,'E']>=-29, 'c2'] = 1
        return torch.tensor(data.iloc[:, 1:].values, dtype=torch.int)

    def get_features(self, data, elements_list):
        for element in elements_list:
            def get_atomic_fraction(comps):
                return comps.get_atomic_fraction(element)
            data[element] = data['composition'].apply(get_atomic_fraction)
        
        return torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)
    
    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int):
        x = self.features[index]
        label = self.labels[index]
        c = self.conditions[index]
        return x, c, label