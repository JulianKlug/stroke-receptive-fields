import os, sys, numpy
sys.path.insert(0, '../')
import analysis.data_loader as dl

# main_dir = '/Users/julian/master/data/from_Server'
# main_dir = '/Users/julian/master/server_output'
main_dir = '/home/klug/data/working_data/'
data_dir = os.path.join(main_dir, 'saved_data')

CLIN, IN, OUT, MASKS = dl.load_saved_data(data_dir)

print('unmasked positive/negative ratio: ', numpy.sum(OUT), OUT.size, numpy.sum(OUT) / OUT.size)

masked_out = OUT[numpy.where(MASKS == 1)]

print('masked positive/negative ratio: ', numpy.sum(masked_out), masked_out.size, numpy.sum(masked_out) / masked_out.size)
