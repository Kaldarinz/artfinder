import artfinder

cr = artfinder.Crossref()
af = artfinder.ArtFinder('aapopov1@mephi.ru')
print(cr.search('Kabashin').get_df())