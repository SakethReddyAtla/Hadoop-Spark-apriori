from mrjob.job import MRJob
from mrjob.step import MRStep
from itertools import combinations

class AprioriMRJob(MRJob):

    def mapper_extract_items(self, _, line):
        # We are splitting all the columns with seperator ','
        fields = line.strip().split(',')
        Invoice = fields[0]  #this is orderid
        Description = fields[2]   # this is style from dataset
        yield Invoice,Description    #return invoice and description

    def reducer_collect_description(self, Invoice, Description):
        Description = ",".join(Description)
        yield Invoice, Description  # returning id with list of items


    def steps(self):
        return [
            MRStep(mapper=self.mapper_extract_items,
                   reducer=self.reducer_collect_description),
        ]

if __name__ == '__main__':
    AprioriMRJob().run()
