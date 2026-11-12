import unittest
import analyze as a


class EfficiencyChecks(unittest.TestCase):
    def row(self, labels=10, mape=9, **kw):
        return dict(actual_train_labels=labels, mean_mape_pct=mape, mean_p90_pct=20,
                    failures=0, learner='rf', method='residual', k=10, **kw)

    def test_minimum_labels_precede_accuracy(self):
        self.assertEqual(a.cheapest([self.row(20, 5), self.row(10, 9)], 10)['actual_train_labels'], 10)

    def test_failure_and_unattained(self):
        bad = self.row(); bad['failures'] = 1
        self.assertIsNone(a.cheapest([bad, self.row(mape=11)], 10))
        self.assertIsNone(a.cheapest([self.row()], 10, 19))

    def test_break_even_boundary(self):
        x = a.breakeven(100, 30, 20)
        self.assertEqual((x['first_nonloss_task_count'], x['first_saving_task_count']), (10, 11))
        self.assertEqual(a.breakeven(0, 30, 20)['first_saving_task_count'], 1)
        self.assertEqual(a.breakeven(100, 20, 30)['status'], 'NO_BREAK_EVEN')

    def test_shared_seed_labels(self):
        base = dict(layer='ffn', dtype='fp32', implementation='standard', validation_group_ids=['v'])
        rows = [dict(base, seed=1, train_group_ids=['a','b']), dict(base, seed=2, train_group_ids=['b','c'])]
        x = a.budget_fields(rows)
        self.assertEqual((x['total_labels_union'], x['raw_records_union']), (4, 12))

    def test_component_labels_do_not_duplicate_validation(self):
        base = dict(layer='conv1', dtype='fp16', seed=1)
        rows = [dict(base, implementation=i, train_group_ids=['a'], validation_group_ids=[])
                for i in a.COMPONENTS['conv1']]
        rows.append(dict(base, implementation='decomposed', train_group_ids=[], validation_group_ids=['v']))
        self.assertEqual(a.budget_fields(rows)['total_labels_union'], 4)


if __name__ == '__main__':
    unittest.main()
