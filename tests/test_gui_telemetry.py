import unittest
from faceswap_gui.telemetry import aggregate_engines

class EngineAggregationTests(unittest.TestCase):
    def test_processes_sum_per_physical_engine(self):
        stem='luid_0x00000000_0x00000111_phys_0'
        result=aggregate_engines([('pid_1_'+stem+'_eng_0_engtype_3D',30),
                                  ('pid_2_'+stem+'_eng_0_engtype_3D',45),
                                  ('pid_1_'+stem+'_eng_1_engtype_Copy',60)])
        self.assertEqual(result[stem],75)
    def test_adapters_separate_and_cap(self):
        a='luid_0x00000000_0x00000111_phys_0';b='luid_0x00000000_0x00000222_phys_0'
        result=aggregate_engines([(a+'_eng_0',150),(b+'_eng_0',25),('invalid',99)])
        self.assertEqual(result,{a:100,b:25})
    def test_missing_is_not_zero(self):self.assertEqual(aggregate_engines([]),{})
