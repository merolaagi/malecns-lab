import json
import re
import unittest
from pathlib import Path
from unittest import mock

import helper
import helper_kb as kb

BASE = Path(__file__).resolve().parent
HTML = {'/': 'index.html', '/3d': 'viewer3d.html'}


def page_file(path):
    return BASE / HTML.get(path, path.strip('/') + '.html')


class KnowledgeTest(unittest.TestCase):
    def test_every_page_in_the_menu_has_notes(self):
        menu = re.findall(r"\['(/[a-z0-9]*)', '", (BASE / 'ink-skin.js').read_text())
        self.assertTrue(menu)
        self.assertEqual(sorted(set(menu) - set(kb.PAGES)), [])

    def test_tour_selectors_exist_on_their_pages(self):
        for path, p in kb.PAGES.items():
            html = page_file(path).read_text()
            for sel, title, _ in p['steps']:
                if not sel: continue
                if sel.startswith('#'): self.assertIn(f'id="{sel[1:]}"', html, f'{path}: {sel}')
                elif sel.startswith('.'): self.assertIn(sel[1:], html, f'{path}: {sel}')

    def test_data_files_named_exist(self):
        for path, p in kb.PAGES.items():
            if p['data']: self.assertTrue((BASE / p['data']).exists(), p['data'])

    def test_proposals_are_complete(self):
        for pr in kb.PROPOSALS:
            for f in helper.FIELDS: self.assertTrue(pr.get(f), (pr['id'], f))
            for pg in pr['pages']: self.assertIn(pg, kb.PAGES)


NO_MODEL = dict(target='llm.active', return_value=None)
CFG = dict(provider='anthropic', kind='anthropic', label='Anthropic (Claude)', base='https://api.anthropic.com', model='m', key='k')


class LocalModeTest(unittest.TestCase):
    def setUp(self):
        self.p = mock.patch('llm.active', return_value=None); self.p.start()

    def tearDown(self): self.p.stop()

    def test_guide(self):
        g = helper.guide('/optogenetics/')
        self.assertEqual(g['title'], 'Virtual optogenetics')
        self.assertIsNone(g['llm'])
        self.assertTrue(any(t['term'] == 'MDN' for t in g['terms']))
        self.assertFalse(helper.guide('/nowhere')['known'])
        json.dumps(g, allow_nan=False)

    def test_definition(self):
        a = helper.ask('What is a Kenyon cell?', '/memory')
        self.assertEqual(a['mode'], 'local')
        self.assertIn('Kenyon cell', a['answer'])
        self.assertIn('sparse', a['answer'])

    def test_measured_and_findings(self):
        a = helper.ask('What is measured and what is assumed here?', '/scaffold')['answer']
        self.assertIn('Measured', a); self.assertIn('Assumed', a)
        self.assertIn('100%', helper.ask('What did the experiments find?', '/scaffold')['answer'])

    def test_pointing_explains_terms(self):
        a = helper.ask('', '/optogenetics', selection='Shuffled wiring and sham controls for the giant fiber')['answer']
        self.assertIn('giant fiber', a); self.assertIn('sham', a)

    def test_search_ranks_the_named_term(self):
        hits = helper.search('ring attractor heading', '/navigation')
        self.assertEqual(hits[0]['title'], 'ring attractor')

    def test_proposals_follow_the_goal(self):
        walk = helper.propose('make the fly walk with coordinated legs', '/optogenetics')
        self.assertEqual(walk['mode'], 'local')
        self.assertIn(walk['proposals'][0]['id'], ('leg-coordination-rules', 'front-leg-route'))
        frame = helper.propose('learn the reference frame for numbers from data', '/scaffold')
        self.assertEqual(frame['proposals'][0]['id'], 'frame-discovery')
        self.assertIn('Hypothesis', helper.build_request(frame['proposals'][0]))

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError): helper.ask('', '/')


class ModelModeTest(unittest.TestCase):
    def patches(self, chat, per_hour=5):
        return [mock.patch('llm.active', return_value=dict(CFG)), mock.patch('llm.chat', chat),
                mock.patch.object(helper, 'LIMIT', helper.Limiter(per_hour, 5)), mock.patch.object(helper, 'VISITOR_LIMIT', helper.Limiter(5, 5))]

    def run_with(self, chat, fn, per_hour=5):
        ps = self.patches(chat, per_hour)
        for p in ps: p.start()
        try: return fn()
        finally:
            for p in ps: p.stop()

    def test_answer_with_page_context(self):
        seen = {}
        def fake(cfg, system, messages, **kw):
            seen.update(cfg=cfg, system=system, messages=messages); return 'An answer.'
        r = self.run_with(fake, lambda: helper.ask('Why?', '/scaffold', history=[{'role': 'assistant', 'text': 'hi'}, {'role': 'user', 'text': 'a'}, {'role': 'assistant', 'text': 'b'}]))
        self.assertEqual((r['mode'], r['model']), ('llm', 'm'))
        self.assertIn('Scaffold learner', seen['system']); self.assertIn('LIVE NUMBERS', seen['system'])
        self.assertEqual([m['role'] for m in seen['messages']], ['user', 'assistant', 'user'])

    def test_visitor_key_is_used_for_that_request(self):
        seen = {}
        def fake(cfg, system, messages, **kw): seen.update(cfg); return 'ok'
        r = self.run_with(fake, lambda: helper.ask('Why?', '/', override={'provider': 'openai', 'model': 'gpt-x', 'key': 'sk-v'}, who='1.2.3.4'))
        self.assertEqual((seen['provider'], seen['key'], seen['base']), ('openai', 'sk-v', 'https://api.openai.com/v1'))
        self.assertEqual(r['mode'], 'llm')
        r = self.run_with(fake, lambda: helper.ask('Why?', '/', override={'provider': 'custom', 'model': 'x', 'key': 'k'}))
        self.assertEqual(r['mode'], 'local')            # no local or custom URLs from a browser key
        self.assertIn('cannot be used', r['note'])

    def test_falls_back_when_unreachable_or_limited(self):
        def boom(*a, **k): raise helper.llm.LLMError('unreachable (down)')
        r = self.run_with(boom, lambda: helper.ask('What is MDN?', '/optogenetics'))
        self.assertEqual(r['mode'], 'local'); self.assertIn('unreachable', r['note']); self.assertIn('backward', r['answer'])
        r = self.run_with(lambda *a, **k: 'x', lambda: helper.ask('What is MDN?', '/', who='9.9.9.9'), per_hour=0)
        self.assertIn('limit', r['note'])
        r = self.run_with(lambda *a, **k: 'x', lambda: helper.ask('What is MDN?', '/', local=True), per_hour=0)
        self.assertEqual(r['mode'], 'llm')              # the owner on the Mac is not rate-limited

    def test_proposals_parsed(self):
        reply = 'Here: [{"title": "T", "idea": "I", "hypothesis": "H", "effort": "S"}]'
        r = self.run_with(lambda *a, **k: reply, lambda: helper.propose('anything', '/scaffold'))
        self.assertEqual(r['mode'], 'llm'); self.assertEqual(r['proposals'][0]['title'], 'T')

    def test_limiter(self):
        lim = helper.Limiter(2, 3)
        self.assertTrue(lim.allow('a', 0)); self.assertTrue(lim.allow('a', 1)); self.assertFalse(lim.allow('a', 2))
        self.assertTrue(lim.allow('b', 3)); self.assertFalse(lim.allow('c', 4))     # daily cap
        self.assertTrue(lim.allow('a', 3700 + 86400))


if __name__ == '__main__':
    unittest.main()
