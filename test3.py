import artfinder

cr = artfinder.Crossref()
print(cr.search('Kabashin').get_df())