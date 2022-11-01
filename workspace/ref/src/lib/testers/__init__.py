from lib.options import options

''' comment out the following line because the name need to be 
replaced '''
# if options.test.name == \
# 'pointnet_jigsaw':
#     from lib.tester.jigsaw_tester import PointNetTester

# if options.test.name == 'PCNEncoderTester':
#     from lib.tester.pcn_encoder_tester import PCNEncoderTester

if options.test.tester == 'PointNetVAETester':
    from lib.testers.tester import PointNetVAETester

if options.test.tester == 'shapenetpart_meshnet2_ssl_mae_test':
    from lib.testers.shapenetpart.meshnet2.ssl_mae_test import Tester
