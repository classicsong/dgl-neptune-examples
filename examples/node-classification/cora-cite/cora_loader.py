from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch as th
import os, sys
from dgl import DGLGraph
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

sys.path.append('../../')
from utils.basic_loader import NodeClassificationDataloader

feature_cols = ['~id', 'Feature1180:int','Feature1181:int','Feature814:int','Feature1182:int','Feature1183:int','Feature813:int','Feature816:int','Feature1184:int','Feature815:int','Feature1185:int','Feature810:int','Feature1186:int','Feature1187:int','Feature812:int','Feature1188:int','Feature1189:int','Feature811:int','Feature818:int','Feature817:int','Feature819:int','Feature830:int','Feature1170:int','Feature1171:int','Feature825:int','Feature1172:int','Feature824:int','Feature1173:int','Feature827:int','Feature1174:int','Feature826:int','Feature1175:int','Feature821:int','Feature820:int','Feature1176:int','Feature823:int','Feature1177:int','Feature822:int','Feature1178:int','Feature1179:int','Feature829:int','Feature828:int','Feature1160:int','Feature1161:int','Feature1162:int','Feature1163:int','Feature1164:int','Feature1165:int','Feature1166:int','Feature1167:int','Feature1168:int','Feature1169:int','Feature1390:int','Feature1391:int','Feature803:int','Feature1150:int','Feature802:int','Feature1392:int','Feature1151:int','Feature805:int','Feature1393:int','Feature804:int','Feature1394:int','Feature1152:int','Feature1395:int','Feature1153:int','Feature1396:int','Feature1154:int','Feature1397:int','Feature1155:int','Feature801:int','Feature1398:int','Feature1156:int','Feature800:int','Feature1399:int','Feature1157:int','Feature1158:int','Feature1159:int','Feature807:int','Feature806:int','Feature809:int','Feature808:int','Feature1380:int','Feature1381:int','Feature1382:int','Feature1140:int','Feature1383:int','Feature1141:int','Feature1142:int','Feature1384:int','Feature1143:int','Feature1385:int','Feature1386:int','Feature1144:int','Feature1387:int','Feature1145:int','Feature1388:int','Feature1146:int','Feature1389:int','Feature1147:int','Feature1148:int','Feature1149:int','Feature1370:int','Feature1371:int','Feature1372:int','Feature1130:int','Feature1373:int','Feature1131:int','Feature1374:int','Feature1132:int','Feature1375:int','Feature1133:int','Feature1134:int','Feature1376:int','Feature1135:int','Feature1377:int','Feature1378:int','Feature1136:int','Feature1379:int','Feature1137:int','Feature1138:int','Feature1139:int','Feature1360:int','Feature1361:int','Feature1362:int','Feature1120:int','Feature1363:int','Feature1121:int','Feature1364:int','Feature1122:int','Feature1365:int','Feature1123:int','Feature1366:int','Feature1124:int','Feature1367:int','Feature1125:int','Feature1126:int','Feature1368:int','Feature1127:int','Feature1369:int','Feature1128:int','Feature1129:int','Feature1350:int','Feature1351:int','Feature1110:int','Feature1352:int','Feature1111:int','Feature1353:int','Feature1354:int','Feature1112:int','Feature1355:int','Feature1113:int','Feature1356:int','Feature1114:int','Feature1357:int','Feature1115:int','Feature1358:int','Feature1116:int','Feature1359:int','Feature1117:int','Feature1118:int','Feature1119:int','Feature182:int','Feature181:int','Feature184:int','Feature183:int','Feature180:int','Feature189:int','Feature186:int','Feature185:int','Feature188:int','Feature187:int','Feature193:int','Feature192:int','Feature195:int','Feature194:int','Feature191:int','Feature190:int','Feature197:int','Feature196:int','Feature199:int','Feature198:int','Feature160:int','Feature162:int','Feature161:int','Feature168:int','Feature167:int','Feature169:int','Feature164:int','Feature163:int','Feature166:int','Feature165:int','Feature171:int','Feature170:int','Feature173:int','Feature172:int','Feature179:int','Feature178:int','Feature175:int','Feature1190:int','Feature174:int','Feature1191:int','Feature177:int','Feature1192:int','Feature176:int','Feature1193:int','Feature1194:int','Feature1195:int','Feature1196:int','Feature1197:int','Feature1198:int','Feature1199:int','Feature261:int','Feature260:int','Feature267:int','Feature266:int','Feature269:int','Feature268:int','Feature263:int','Feature90:int','Feature262:int','Feature92:int','Feature265:int','Feature91:int','Feature264:int','Feature94:int','Feature93:int','Feature96:int','Feature95:int','Feature98:int','Feature259:int','Feature97:int','Feature99:int','Feature270:int','Feature272:int','Feature271:int','Feature278:int','Feature277:int','Feature279:int','Feature274:int','Feature273:int','Feature276:int','Feature81:int','Feature275:int','Feature80:int','Feature83:int','Feature82:int','Feature85:int','Feature84:int','Feature87:int','Feature86:int','Feature89:int','Feature88:int','Feature481:int','Feature480:int','Feature487:int','Feature245:int','Feature486:int','Feature244:int','Feature247:int','Feature489:int','Feature246:int','Feature488:int','Feature483:int','Feature241:int','Feature482:int','Feature240:int','Feature70:int','Feature485:int','Feature243:int','Feature484:int','Feature242:int','Feature72:int','Feature71:int','Feature74:int','Feature73:int','Feature238:int','Feature76:int','Feature479:int','Feature237:int','Feature75:int','Feature78:int','Feature239:int','Feature77:int','Feature79:int','Feature490:int','Feature492:int','Feature250:int','Feature491:int','Feature498:int','Feature256:int','Feature255:int','Feature497:int','Feature258:int','Feature499:int','Feature257:int','Feature494:int','Feature252:int','Feature493:int','Feature251:int','Feature254:int','Feature496:int','Feature495:int','Feature253:int','Feature61:int','Feature60:int','Feature63:int','Feature62:int','Feature65:int','Feature249:int','Feature64:int','Feature248:int','Feature67:int','Feature66:int','Feature69:int','Feature68:int','Feature223:int','Feature465:int','Feature222:int','Feature464:int','Feature467:int','Feature225:int','Feature466:int','Feature224:int','Feature461:int','Feature460:int','Feature463:int','Feature221:int','Feature462:int','Feature220:int','Feature219:int','Feature458:int','Feature216:int','Feature215:int','Feature699:int','Feature457:int','Feature218:int','Feature459:int','Feature217:int','Feature5:int','Feature4:int','Feature3:int','Feature2:int','Feature1:int','Feature0:int','Feature470:int','Feature9:int','Feature8:int','Feature7:int','Feature6:int','Feature476:int','Feature234:int','Feature475:int','Feature233:int','Feature478:int','Feature236:int','Feature477:int','Feature235:int','Feature230:int','Feature472:int','Feature471:int','Feature474:int','Feature232:int','Feature231:int','Feature473:int','Feature469:int','Feature227:int','Feature468:int','Feature226:int','Feature229:int','Feature228:int','Feature685:int','Feature443:int','Feature201:int','Feature684:int','Feature442:int','Feature200:int','Feature687:int','Feature445:int','Feature203:int','Feature686:int','Feature444:int','Feature202:int','Feature681:int','Feature680:int','Feature683:int','Feature441:int','Feature682:int','Feature440:int','Feature439:int','Feature678:int','Feature436:int','Feature677:int','Feature435:int','Feature438:int','Feature679:int','Feature437:int','Feature690:int','Feature454:int','Feature212:int','Feature696:int','Feature695:int','Feature453:int','Feature211:int','Feature214:int','Feature698:int','Feature456:int','Feature455:int','Feature213:int','Feature697:int','Feature692:int','Feature450:int','Feature691:int','Feature694:int','Feature452:int','Feature210:int','Feature693:int','Feature451:int','Feature209:int','Feature208:int','Feature447:int','Feature205:int','Feature689:int','Feature446:int','Feature204:int','Feature688:int','Feature207:int','Feature449:int','Feature206:int','Feature448:int','Feature663:int','Feature421:int','Feature662:int','Feature420:int','Feature423:int','Feature665:int','Feature422:int','Feature664:int','Feature661:int','Feature660:int','Feature418:int','Feature659:int','Feature417:int','Feature419:int','Feature414:int','Feature898:int','Feature656:int','Feature655:int','Feature1220:int','Feature413:int','Feature897:int','Feature1221:int','Feature658:int','Feature416:int','Feature1222:int','Feature415:int','Feature899:int','Feature657:int','Feature1223:int','Feature1224:int','Feature1225:int','Feature1226:int','Feature1227:int','Feature1228:int','Feature1229:int','Feature1209:int','Feature674:int','Feature432:int','Feature431:int','Feature673:int','Feature676:int','Feature434:int','Feature675:int','Feature433:int','Feature670:int','Feature430:int','Feature672:int','Feature671:int','Feature429:int','Feature428:int','Feature667:int','Feature425:int','Feature666:int','Feature424:int','Feature669:int','Feature1210:int','Feature427:int','Feature668:int','Feature426:int','Feature1211:int','Feature1212:int','Feature1213:int','Feature1214:int','Feature1215:int','Feature1216:int','Feature1217:int','Feature1218:int','Feature1219:int','Feature883:int','Feature641:int','Feature882:int','Feature640:int','Feature885:int','Feature643:int','Feature401:int','Feature884:int','Feature642:int','Feature400:int','Feature881:int','Feature880:int','Feature638:int','Feature879:int','Feature637:int','Feature639:int','Feature876:int','Feature634:int','Feature875:int','Feature633:int','Feature878:int','Feature636:int','Feature877:int','Feature635:int','Feature1200:int','Feature1201:int','Feature1202:int','Feature1203:int','Feature1204:int','Feature1205:int','Feature1206:int','Feature1207:int','Feature1208:int','Feature1429:int','Feature894:int','Feature652:int','Feature410:int','Feature893:int','Feature651:int','Feature654:int','Feature412:int','Feature896:int','Feature895:int','Feature653:int','Feature411:int','Feature890:int','Feature892:int','Feature650:int','Feature891:int','Feature407:int','Feature649:int','Feature406:int','Feature648:int','Feature409:int','Feature408:int','Feature887:int','Feature645:int','Feature403:int','Feature886:int','Feature644:int','Feature402:int','Feature1430:int','Feature647:int','Feature405:int','Feature889:int','Feature646:int','Feature1431:int','Feature404:int','Feature888:int','Feature1432:int','Feature1418:int','Feature1419:int','Feature861:int','Feature860:int','Feature863:int','Feature621:int','Feature862:int','Feature620:int','Feature858:int','Feature616:int','Feature615:int','Feature857:int','Feature618:int',
'Feature859:int','Feature617:int','Feature854:int','Feature612:int','Feature853:int','Feature611:int','Feature614:int','Feature856:int','Feature855:int','Feature1420:int','Feature613:int','Feature1421:int','Feature1422:int','Feature1423:int','Feature1424:int','Feature1425:int','Feature1426:int','Feature619:int','Feature1427:int','Feature1428:int','Feature1407:int','Feature1408:int','Feature1409:int','Feature630:int','Feature872:int','Feature871:int','Feature874:int','Feature632:int','Feature631:int','Feature873:int','Feature870:int','Feature869:int','Feature627:int','Feature868:int','Feature626:int','Feature629:int','Feature628:int','Feature623:int','Feature865:int','Feature622:int','Feature864:int','Feature867:int','Feature625:int','Feature866:int','Feature624:int','Feature1410:int','Feature1411:int','Feature1412:int','Feature1413:int','Feature1414:int','Feature1415:int','Feature1416:int','Feature1417:int','Feature841:int','Feature840:int','Feature836:int','Feature835:int','Feature838:int','Feature837:int','Feature832:int','Feature831:int','Feature834:int','Feature833:int','Feature1400:int','Feature1401:int','Feature1402:int','Feature1403:int','Feature839:int','Feature1404:int','Feature1405:int','Feature1406:int','Feature850:int','Feature852:int','Feature610:int','Feature851:int',
'Feature847:int','Feature605:int','Feature846:int','Feature604:int','Feature607:int','Feature849:int','Feature606:int','Feature848:int','Feature843:int','Feature601:int','Feature842:int','Feature600:int','Feature845:int','Feature603:int','Feature844:int','Feature602:int','Feature609:int','Feature608:int','Feature940:int','Feature1060:int','Feature935:int','Feature1061:int','Feature934:int','Feature1062:int','Feature1063:int','Feature937:int','Feature936:int','Feature1064:int','Feature931:int','Feature1065:int','Feature930:int','Feature1066:int','Feature933:int','Feature1067:int','Feature932:int','Feature1068:int','Feature1069:int','Feature939:int','Feature938:int','Feature951:int','Feature950:int','Feature1290:int','Feature1291:int','Feature1292:int','Feature1050:int','Feature946:int','Feature704:int','Feature703:int','Feature1293:int','Feature1051:int','Feature945:int','Feature1294:int','Feature1052:int','Feature948:int','Feature706:int','Feature1295:int','Feature1053:int','Feature947:int','Feature705:int','Feature1054:int','Feature942:int','Feature700:int','Feature1296:int','Feature1055:int','Feature941:int','Feature1297:int','Feature702:int','Feature1298:int','Feature1056:int','Feature944:int','Feature943:int','Feature701:int','Feature1299:int','Feature1057:int','Feature1058:int','Feature1059:int','Feature708:int','Feature949:int','Feature707:int','Feature709:int','Feature1280:int','Feature913:int','Feature1281:int','Feature1282:int','Feature1040:int','Feature912:int','Feature915:int','Feature1283:int','Feature1041:int','Feature1284:int','Feature1042:int','Feature914:int','Feature1285:int','Feature1043:int','Feature1286:int','Feature1044:int','Feature911:int','Feature1287:int','Feature1045:int','Feature1046:int','Feature910:int','Feature1288:int','Feature1047:int','Feature1289:int','Feature1048:int','Feature1049:int','Feature917:int','Feature916:int','Feature919:int','Feature918:int','Feature1270:int','Feature924:int','Feature1271:int','Feature923:int','Feature1030:int','Feature926:int','Feature1272:int','Feature1031:int','Feature925:int','Feature1273:int','Feature1274:int','Feature1032:int','Feature920:int','Feature1275:int','Feature1033:int','Feature1276:int','Feature1034:int','Feature922:int','Feature1277:int','Feature1035:int','Feature921:int','Feature1278:int','Feature1036:int','Feature1279:int','Feature1037:int','Feature1038:int','Feature1039:int','Feature928:int','Feature927:int','Feature929:int','Feature1260:int','Feature1261:int','Feature1262:int','Feature1020:int','Feature1263:int','Feature1021:int','Feature1022:int','Feature1264:int','Feature1023:int','Feature1265:int','Feature1266:int','Feature1024:int','Feature1267:int','Feature1025:int','Feature1268:int','Feature1026:int','Feature1269:int','Feature1027:int','Feature1028:int','Feature1029:int','Feature902:int','Feature901:int','Feature1250:int','Feature904:int','Feature903:int','Feature1251:int','Feature1252:int','Feature1010:int','Feature1253:int','Feature1011:int','Feature1254:int','Feature1012:int','Feature900:int','Feature1255:int','Feature1013:int','Feature1014:int','Feature1256:int','Feature1015:int','Feature909:int','Feature1257:int','Feature1258:int','Feature1016:int','Feature1259:int','Feature1017:int','Feature1018:int','Feature906:int','Feature1019:int','Feature905:int','Feature908:int','Feature907:int','Feature1240:int','Feature1241:int','Feature1242:int','Feature1000:int','Feature1243:int','Feature1001:int','Feature1244:int','Feature1002:int','Feature1245:int','Feature1003:int','Feature1246:int','Feature1004:int','Feature1247:int','Feature1005:int','Feature1006:int','Feature1248:int','Feature1007:int','Feature1249:int','Feature1008:int','Feature1009:int','Feature1230:int','Feature1231:int','Feature1232:int','Feature1233:int','Feature1234:int','Feature1235:int','Feature1236:int','Feature1237:int','Feature1238:int','Feature1239:int','Feature10:int','Feature12:int','Feature11:int','Feature14:int','Feature13:int','Feature16:int','Feature15:int','Feature18:int','Feature17:int','Feature19:int','Feature50:int','Feature52:int','Feature51:int','Feature54:int','Feature53:int','Feature56:int','Feature55:int','Feature58:int','Feature57:int','Feature59:int','Feature1090:int','Feature1091:int','Feature1092:int','Feature1093:int','Feature1094:int','Feature1095:int','Feature1096:int','Feature41:int','Feature40:int','Feature1097:int','Feature1098:int','Feature43:int','Feature42:int','Feature1099:int','Feature45:int','Feature44:int','Feature47:int','Feature46:int','Feature49:int','Feature48:int','Feature281:int','Feature280:int','Feature283:int','Feature282:int','Feature289:int','Feature288:int','Feature285:int','Feature284:int','Feature1080:int','Feature287:int','Feature1081:int','Feature286:int','Feature1082:int','Feature1083:int','Feature1084:int','Feature30:int','Feature1085:int','Feature1086:int','Feature1087:int','Feature32:int','Feature31:int','Feature1088:int','Feature34:int','Feature1089:int','Feature33:int','Feature36:int','Feature35:int','Feature38:int','Feature37:int','Feature39:int','Feature292:int','Feature291:int','Feature294:int','Feature293:int','Feature290:int','Feature299:int','Feature296:int','Feature295:int','Feature1070:int','Feature298:int','Feature1071:int','Feature297:int','Feature1072:int','Feature1073:int','Feature1074:int','Feature1075:int','Feature1076:int','Feature21:int','Feature20:int','Feature1077:int','Feature1078:int','Feature23:int','Feature22:int','Feature1079:int','Feature25:int','Feature24:int','Feature27:int','Feature26:int','Feature29:int','Feature28:int','Feature380:int','Feature382:int','Feature140:int','Feature381:int','Feature388:int','Feature146:int','Feature387:int','Feature145:int','Feature148:int','Feature389:int','Feature147:int','Feature142:int','Feature384:int','Feature383:int','Feature141:int','Feature386:int','Feature144:int','Feature143:int','Feature385:int','Feature139:int','Feature138:int','Feature391:int','Feature390:int','Feature151:int','Feature393:int','Feature150:int','Feature392:int','Feature399:int','Feature157:int','Feature398:int','Feature156:int','Feature159:int','Feature158:int','Feature395:int','Feature153:int','Feature394:int','Feature152:int','Feature397:int','Feature155:int','Feature396:int','Feature154:int','Feature149:int','Feature360:int','Feature366:int','Feature124:int','Feature365:int','Feature123:int','Feature126:int','Feature368:int','Feature367:int','Feature125:int','Feature362:int','Feature120:int','Feature361:int','Feature364:int','Feature122:int','Feature363:int','Feature121:int','Feature359:int','Feature117:int','Feature358:int','Feature116:int','Feature119:int','Feature118:int','Feature371:int','Feature370:int','Feature135:int','Feature377:int','Feature134:int','Feature376:int','Feature379:int','Feature137:int','Feature378:int','Feature136:int','Feature373:int','Feature131:int','Feature372:int','Feature130:int','Feature375:int','Feature133:int','Feature374:int','Feature132:int','Feature128:int','Feature127:int','Feature369:int','Feature129:int','Feature580:int','Feature102:int','Feature586:int','Feature344:int','Feature343:int','Feature101:int','Feature585:int','Feature588:int','Feature346:int','Feature104:int','Feature103:int','Feature587:int','Feature345:int','Feature582:int','Feature340:int','Feature581:int','Feature342:int','Feature100:int','Feature584:int','Feature583:int','Feature341:int','Feature579:int','Feature337:int','Feature578:int','Feature336:int','Feature339:int','Feature338:int','Feature591:int','Feature590:int','Feature597:int','Feature355:int','Feature113:int','Feature596:int','Feature354:int','Feature112:int','Feature599:int','Feature357:int','Feature115:int','Feature598:int','Feature356:int','Feature114:int','Feature351:int','Feature593:int','Feature350:int','Feature592:int','Feature111:int','Feature595:int','Feature353:int','Feature110:int','Feature594:int','Feature352:int','Feature109:int','Feature348:int','Feature106:int','Feature589:int','Feature347:int','Feature105:int','Feature108:int','Feature349:int','Feature107:int','Feature564:int','Feature322:int','Feature563:int','Feature321:int','Feature566:int','Feature324:int','Feature565:int','Feature323:int','Feature560:int','Feature562:int','Feature320:int','Feature561:int','Feature319:int','Feature318:int','Feature799:int','Feature557:int','Feature315:int','Feature798:int','Feature556:int','Feature314:int','Feature559:int','Feature317:int','Feature558:int','Feature316:int','Feature575:int','Feature333:int','Feature574:int','Feature332:int','Feature335:int','Feature577:int','Feature334:int','Feature576:int','Feature571:int','Feature570:int','Feature573:int','Feature331:int','Feature572:int','Feature330:int','Feature329:int','Feature326:int','Feature568:int','Feature567:int','Feature325:int','Feature328:int','Feature327:int','Feature569:int','Feature542:int','Feature300:int','Feature784:int','Feature783:int','Feature541:int','Feature302:int','Feature786:int','Feature544:int','Feature543:int','Feature301:int','Feature785:int','Feature780:int','Feature782:int','Feature540:int','Feature781:int','Feature539:int','Feature538:int','Feature535:int','Feature1340:int','Feature777:int','Feature534:int','Feature1341:int','Feature776:int','Feature1342:int','Feature1100:int','Feature779:int','Feature537:int','Feature1343:int','Feature1101:int','Feature778:int','Feature536:int','Feature1102:int','Feature1344:int','Feature1103:int','Feature1345:int','Feature1346:int','Feature1104:int','Feature1347:int','Feature1105:int','Feature1348:int','Feature1106:int','Feature1349:int','Feature1107:int','Feature1108:int','Feature1109:int','Feature311:int','Feature795:int','Feature553:int','Feature310:int','Feature794:int','Feature552:int','Feature797:int','Feature555:int','Feature313:int','Feature796:int','Feature554:int','Feature312:int','Feature791:int','Feature790:int','Feature551:int','Feature793:int','Feature550:int','Feature792:int','Feature308:int','Feature549:int','Feature307:int','Feature309:int',
'Feature788:int','Feature546:int','Feature304:int','Feature303:int','Feature1330:int','Feature787:int','Feature545:int','Feature548:int','Feature306:int','Feature1331:int','Feature1332:int','Feature789:int','Feature547:int','Feature305:int','Feature1333:int','Feature1334:int','Feature1335:int','Feature1336:int','Feature1337:int','Feature1338:int','Feature1339:int','Feature1319:int','Feature762:int','Feature520:int','Feature761:int','Feature764:int','Feature522:int','Feature763:int','Feature521:int','Feature760:int','Feature759:int','Feature517:int','Feature758:int','Feature516:int','Feature519:int','Feature518:int','Feature997:int','Feature755:int','Feature513:int','Feature996:int','Feature754:int','Feature512:int','Feature999:int','Feature757:int','Feature515:int','Feature1320:int','Feature998:int','Feature756:int','Feature514:int','Feature1321:int','Feature1322:int','Feature1323:int','Feature1324:int','Feature1325:int','Feature1326:int','Feature1327:int','Feature1328:int','Feature1329:int','Feature1308:int','Feature1309:int','Feature773:int','Feature531:int','Feature772:int','Feature530:int','Feature775:int','Feature533:int','Feature774:int','Feature532:int','Feature771:int','Feature770:int','Feature528:int','Feature527:int','Feature769:int','Feature529:int','Feature766:int','Feature524:int',
'Feature765:int','Feature523:int','Feature526:int','Feature768:int','Feature1310:int','Feature767:int','Feature525:int','Feature1311:int','Feature1312:int','Feature1313:int','Feature1314:int','Feature1315:int','Feature1316:int','Feature1317:int','Feature1318:int','Feature982:int','Feature740:int','Feature981:int','Feature742:int','Feature500:int','Feature984:int','Feature983:int','Feature741:int','Feature980:int','Feature979:int','Feature737:int','Feature978:int','Feature736:int','Feature739:int','Feature738:int','Feature975:int','Feature733:int','Feature974:int','Feature732:int','Feature735:int','Feature977:int','Feature734:int','Feature976:int','Feature1300:int','Feature1301:int','Feature1302:int','Feature1303:int','Feature1304:int','Feature1305:int','Feature1306:int','Feature1307:int','Feature751:int','Feature993:int','Feature750:int','Feature992:int','Feature511:int','Feature995:int','Feature753:int','Feature510:int','Feature994:int','Feature752:int','Feature991:int','Feature990:int','Feature748:int','Feature506:int','Feature989:int','Feature747:int','Feature505:int','Feature508:int','Feature749:int','Feature507:int','Feature502:int','Feature986:int','Feature744:int','Feature743:int','Feature501:int','Feature985:int','Feature988:int','Feature746:int','Feature504:int','Feature503:int','Feature987:int','Feature745:int','Feature509:int','Feature960:int','Feature962:int','Feature720:int','Feature961:int','Feature957:int','Feature715:int','Feature956:int','Feature714:int','Feature959:int','Feature717:int','Feature958:int','Feature716:int','Feature711:int','Feature953:int','Feature710:int','Feature952:int','Feature955:int','Feature713:int','Feature954:int','Feature712:int','Feature719:int','Feature718:int','Feature971:int','Feature970:int','Feature973:int','Feature731:int','Feature972:int','Feature730:int','Feature726:int','Feature968:int','Feature967:int','Feature725:int','Feature728:int','Feature727:int','Feature969:int','Feature964:int','Feature722:int','Feature963:int','Feature721:int','Feature966:int','Feature724:int','Feature965:int','Feature723:int','Feature729:int']

class NeptuneCoraDataset(NodeClassificationDataloader):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.

    The data is from Neptune database
    """
    def __init__(self, device, self_loop=True, valid_ratio=0.1, test_ratio=0.2):
        super(NeptuneCoraDataset, self).__init__('cora')

        # Step 1: load feature for the graph and build id mapping
        # we ignore the label row, data is stored as '~id ~label feats ...'
        self._load_onehot_feature([("~/data/cora/1582098279283/nodes/paper-1.csv",',', feature_cols)], device)
        # Step 2: load labels
        # we ignore the label row, data is stored as '~id ~label feats ...'
        self._load_raw_label([("~/data/cora/1582098279283/nodes/paper-1.csv", ',', ['~id', 'category:string'])])
        # Step 3: load graph
        # we ignore the label row, data is streod as '~edge_id ~edge_label ~from ~to', we use from and to here
        self._load_raw_graph([(None, "~/data/cora/1582098279283/edges/edge-1.csv",',', ['~from', '~to'])])
        # Step 4: build graph
        self._build_graph(self_loop, symmetric=True)
        # Step 5: load node feature
        self._load_node_feature(device)
        # Step 6: Split labels
        self._split_labels(device, valid_ratio, test_ratio)

        self._n_classes = len(self._labels[0].label_map)
        n_edges = self._g.number_of_edges()
        print("""----Data statistics------'
        #Edges %d
        #Classes %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
            (n_edges, self._n_classes,
                self._train_set[0].shape[0],
                self._valid_set[0].shape[0],
                self._test_set[0].shape[0]))

    @property
    def n_class(self):
        return self._n_classes

class CoraDataset(NodeClassificationDataloader):
    r"""Cora citation network dataset. Nodes mean author and edges mean citation
    relationships.
    """
    def __init__(self, device, self_loop=True, valid_ratio=0.1, test_ratio=0.2):
        super(CoraDataset, self).__init__('cora')
        self._download_data()
        # Step 1: load feature for the graph and build id mapping
        self._load_onehot_feature([("{}/cora/cora.content".format(self._dir),'\t', [0, (1,-1)], (0,0))], device)
        # Step 2: load labels
        self._load_raw_label([("{}/cora/cora.content".format(self._dir),'\t', [0, -1], (0,0))])
        # Step 3: load graph
        self._load_raw_graph([(None, "{}/cora/cora.cites".format(self._dir),'\t', [0, 1], (0,0))])
        # Step 4: build graph
        self._build_graph(self_loop, symmetric=True)
        # Step 5: load node feature
        self._load_node_feature(device)
        # Step 6: Split labels
        self._split_labels(device, valid_ratio, test_ratio)

        self._n_classes = len(self._labels[0].label_map)
        n_edges = self._g.number_of_edges()
        print("""----Data statistics------'
        #Edges %d
        #Classes %d
        #Train samples %d
        #Val samples %d
        #Test samples %d""" %
            (n_edges, self._n_classes,
                self._train_set[0].shape[0],
                self._valid_set[0].shape[0],
                self._test_set[0].shape[0]))

    def _download_data(self):
        self._dir = get_download_dir()
        zip_file_path='{}/{}.zip'.format(self._dir, self._name)
        download(_get_dgl_url("dataset/cora_raw.zip"), path=zip_file_path)
        extract_archive(zip_file_path,
                        '{}/{}'.format(self._dir, self._name))

    def _split_labels(self, device, valid_ratio=0.1, test_ratio=0.2):
        ids, labels = self._labels[0].id_labels
        ids = th.LongTensor(ids).to(device)
        labels = th.LongTensor(labels).to(device)
        train_idx = range(140)
        valid_idx = range(200, 500)
        test_idx = range(500, 1500)

        self._test_set = (ids[test_idx], labels[test_idx])
        self._valid_set = (ids[valid_idx], labels[valid_idx])
        self._train_set = (ids[train_idx], labels[train_idx])

    @property
    def n_class(self):
        return self._n_classes