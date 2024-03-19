# FILEPATH: /c:/Users/slluser/Documents/python_data_science/data science assignment 1/test_data_science_assignment.py
import unittest
import pandas as pd
from data_science_assignment import *




class TestDataScienceAssignment(unittest.TestCase):
    def setUp(self):
        self.data = pd.read_csv('covid_data.csv', low_memory=False)

    def test_data_loaded(self):
        self.assertIsNotNone(self.data)

    def test_vaccination_coverage_percent(self):
        self.data['vaccination_coverage_percent'] = (self.data['people_vaccinated'] / self.data['population']) * 100
        self.assertIn('vaccination_coverage_percent', self.data.columns)

    def test_date_manipulation(self):
        self.data['date'] = pd.to_datetime(self.data['date']).apply(lambda x: x.date())
        self.data['year'] = self.data['date'].apply(lambda x: x.year)
        self.data['month'] = self.data['date'].apply(lambda x: x.month)
        self.data['day'] = self.data['date'].apply(lambda x: x.day)
        self.assertIn('year', self.data.columns)
        self.assertIn('month', self.data.columns)
        self.assertIn('day', self.data.columns)

    def test_description_statistics(self):
        desc_stats = self.data[self.data['year'] == 2021][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe()
        self.assertIsNotNone(desc_stats)

    def test_description_statistics_ireland(self):
        desc_stats_ireland = self.data[self.data['location'] == 'Ireland'][['total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].describe()
        self.assertIsNotNone(desc_stats_ireland)

if __name__ == '__main__':
    unittest.main()