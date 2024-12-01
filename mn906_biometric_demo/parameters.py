### Modifs 2021may27, V5
##########################

# Face detection theshold , between 0 and 1; 
##############################################
# the higher the more confident that the detected object is a face 
face_confidence_threshold = 0.8

# Face verification threshold, between 0 and 1; 
###############################################
# Higher threshold for higher security
threshold = 0.55

# path to the Face recognition DNN
####################################
face_reco_model = './models/face_reco-model.t7'

# Previous version enrolment was done with 10 images automatically chosen
# for V5: enrollement is done with the space bar
# leave the command anway
number_of_templates = 10


# Continous enrolment
# dij 2022jan18 , test with True
# cont_enroll = False
cont_enroll = True
# If continuous enrolment is set to True; verification templates which distance 
# is between 2 thresholds 'lower_enrolment_update_threshold' and 'upper_enrolment_update_threshold' 
# are added to the enrolment database
lower_enrolment_update_threshold = 0.67
upper_enrolment_update_threshold = 0.75
