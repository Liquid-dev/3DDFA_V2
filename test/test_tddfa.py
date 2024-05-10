from tddfa import TDDFA
from tddfa.utils import pose
import cv2
tddfa_detector = TDDFA.TDDFA(is_default=True)
def test_compatibility():
    image = cv2.imread("test/1200px-Mona_Lisa_detail_face.jpg")
    param_lst, _ =  tddfa_detector(image, [[188, 273, 850, 1200]])
    _, tp = pose.calc_pose(param_lst[0])

    py37_result = [21.043000867989004, 6.444989611717696, -0.43137103355505546]

    for py37, p in zip(py37_result, tp):
        assert abs(py37 - p) < 0.01

