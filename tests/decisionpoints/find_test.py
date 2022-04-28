import unittest
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.objects.petri_net.obj import PetriNet
from pm4py.objects.petri_net.utils import petri_utils
from exdpn.decisionpoints import find_decision_points
from importlib.resources import read_text

class TestFindDecisionPoints(unittest.TestCase):
    def test_simple_net(self):
            net, real_decision_points = make_simple_net()
            decision_points = find_decision_points(net)

            # Detected decision points are as expected
            self.assertEqual(decision_points, real_decision_points)

    def test_complex_net(self):
            net, real_decision_points = make_complex_net()
            decision_points = find_decision_points(net)

            # Detected decision points are as expected
            self.assertEqual(decision_points, real_decision_points)


def make_simple_net():
    #               -> [b] 
    # o -> [a] -> o        -> o
    #               -> [c]
    #
    places = [
        PetriNet.Place('start'),
        PetriNet.Place('choice'),
        PetriNet.Place('end')
    ]
    start, choice, end = places

    transitions= [
        PetriNet.Transition('a','a'),
        PetriNet.Transition('b','b'),
        PetriNet.Transition('c','c')
    ]
    a, b, c = transitions
    
    net = PetriNet("Testing Petri net", places=places, transitions=transitions)

    petri_utils.add_arc_from_to(start,a,net)
    
    petri_utils.add_arc_from_to(a,choice, net)
    
    petri_utils.add_arc_from_to(choice,b, net)
    petri_utils.add_arc_from_to(choice,c, net)
    petri_utils.add_arc_from_to(b,end, net)
    petri_utils.add_arc_from_to(c,end, net)

    decision_points = {
        choice: {b,c}
    }

    return net, decision_points

def make_complex_net():
    places = [
        PetriNet.Place("p1"),
        PetriNet.Place("p2"),
        PetriNet.Place("p3"),
        PetriNet.Place("p4")
    ]
    p1, p2, p3, p4 = places

    transitions = [
        PetriNet.Transition("a","a"),
        PetriNet.Transition("b","b"),
        PetriNet.Transition("c","c"),
        PetriNet.Transition("d","d"),
        PetriNet.Transition("e","e"),
        PetriNet.Transition("f","f"),
    ]
    a, b, c, d, e, f = transitions

    net = PetriNet("Testing Petri net", places=places, transitions=transitions)

    petri_utils.add_arc_from_to(p1 ,a ,net)

    petri_utils.add_arc_from_to(a, p2 ,net)
    petri_utils.add_arc_from_to(a, p3 ,net)

    petri_utils.add_arc_from_to(p2, b ,net)
    petri_utils.add_arc_from_to(p2, c ,net)
    petri_utils.add_arc_from_to(p2, d ,net)

    petri_utils.add_arc_from_to(p3, e,net)
    petri_utils.add_arc_from_to(p3, f,net)

    petri_utils.add_arc_from_to(b, p4,net)
    petri_utils.add_arc_from_to(c, p4,net)
    petri_utils.add_arc_from_to(d, p4,net)
    petri_utils.add_arc_from_to(e, p4,net)
    petri_utils.add_arc_from_to(f, p4,net)
    
    petri_utils.add_arc_from_to(f, p3,net)

    decision_points = {
        p2: {b,c,d},
        p3: {e,f}
    }

    return net, decision_points

