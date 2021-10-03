from unittest import TestCase

import numpy as np

from drug_interactions.reader.dal import Drug, DrugBank
from drug_interactions.training.train import Trainer
from drug_interactions.datasets.OneHotSmilesDataset import OneHotSmilesDrugDataset

class OneHotSmilesDrugDatasetTestCase(TestCase):

    def setUp(self):

        self.Olddrug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        self.Olddrug2 = Drug(**{'id_': '2', 'interactions': {('3', 'baz')}})
        self.Olddrug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar'), ('2', 'baz')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
  

        self.Newdrug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar'), ('4', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        self.Newdrug2 = Drug(**{'id_': '2', 'interactions': {('3', 'baz'), ('5', 'bar')}})
        self.Newdrug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar'), ('2', 'baz')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        self.Newdrug4 = Drug(**{'id_': '4', 'interactions': {('1', 'bar'), ('6', 'goo')}})
        self.Newdrug5 = Drug(**{'id_': '5', 'interactions': {('2', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        self.Newdrug6 = Drug(**{'id_': '6', 'interactions': {('4', 'goo')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})

        self.oldDrugBank = DrugBank('9', [self.Olddrug1, self.Olddrug2, self.Olddrug3])

        self.NewDrugBank = DrugBank('12', [self.Newdrug1, self.Newdrug2, self.Newdrug3, self.Newdrug4, self.Newdrug5, self.Newdrug6])

        self.dataset = OneHotSmilesDrugDataset(self.oldDrugBank, self.NewDrugBank, neg_pos_ratio=1.0, atom_size=300, atom_info=21, struct_info=21)


    def testGetSmilesDrugsOld(self):

        Olddrug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        Olddrug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        oldDrugBank = DrugBank('9', [Olddrug1, Olddrug3])

        old_drug_bank = self.dataset.get_smiles_drugs(self.dataset.old_drug_bank)

        self.assertEqual(old_drug_bank, oldDrugBank)

    
    def testGetSmilesDrugsNew(self):

        Newdrug1 = Drug(**{'id_': '1', 'interactions': {('3', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        Newdrug3 = Drug(**{'id_': '3', 'interactions': {('1', 'bar')}, 'smiles': 'CC(=O)NC1=CC=C(O)C=C1'})
        newDrugBank = DrugBank('9', [Newdrug1, Newdrug3])

        new_drug_bank = self.dataset.get_smiles_drugs(self.dataset.new_drug_bank)

        self.assertEqual(new_drug_bank, newDrugBank)

    
    def testGetSmilesFeature(self):
        drug_to_smiles = {'1': "CC(=O)", '2': "CC(=H)"}

        drug_to_smiles_features_gt = {'1': np.array([[1, 0, 0, 0, 0, 0 ,0],
                                                     [0, 0, 0, 1, 0, 0 ,0],
                                                     [0, 1, 0, 0, 0, 0 ,0],
                                                     [0, 0, 0, 0, 0, 0 ,1],
                                                     [0, 0, 0, 0, 1, 0 ,0]]),
                                      '2': np.array([[1, 0, 0, 0, 0, 0 ,0],
                                                     [0, 0, 0, 1, 0, 0 ,0],
                                                     [0, 1, 0, 0, 0, 0 ,0],
                                                     [0, 0, 0, 0, 0, 1 ,0],
                                                     [0, 0, 0, 0, 1, 0 ,0]])}

        drug_to_smiles_features = self.dataset.get_smiles_features(drug_to_smiles, {'2': "CC(=H)"})
        print(drug_to_smiles_features)
        self.assertCountEqual(drug_to_smiles_features, drug_to_smiles_features_gt)
