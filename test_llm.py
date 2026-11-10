import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import llm


class Reply:
    def __init__(self, obj): self.obj = obj
    def read(self): return json.dumps(self.obj).encode()
    def __enter__(self): return self
    def __exit__(self, *a): return False


class SettingsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(); d = Path(self.dir.name)
        self.ps = [mock.patch.object(llm, 'SETTINGS', d / 'llm.json'), mock.patch.object(llm, 'OLD_KEY', d / 'anthropic-key'),
                   mock.patch.object(llm, 'CONFIG_DIR', d), mock.patch.dict(os.environ, {}, clear=False)]
        for p in self.ps: p.start()
        for k in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'GEMINI_API_KEY', 'OPENROUTER_API_KEY'): os.environ.pop(k, None)

    def tearDown(self):
        for p in self.ps: p.stop()
        self.dir.cleanup()

    def test_nothing_configured(self):
        self.assertIsNone(llm.active())
        self.assertIsNone(llm.view(False)['active'])

    def test_save_choose_and_mask(self):
        llm.update(dict(provider='openai', models={'openai': 'gpt-x'}, keys={'openai': 'sk-secret-1234'}))
        self.assertEqual(stat.S_IMODE(os.stat(llm.SETTINGS).st_mode), 0o600)
        cfg = llm.active()
        self.assertEqual((cfg['provider'], cfg['model'], cfg['key']), ('openai', 'gpt-x', 'sk-secret-1234'))
        v = llm.view(True)
        self.assertEqual(v['settings']['keys']['openai'], '…1234')
        self.assertNotIn('sk-secret', json.dumps(v)); self.assertNotIn('sk-secret', json.dumps(llm.view(False)))
        self.assertNotIn('settings', llm.view(False))
        llm.update(dict(keys={'openai': ''}))
        self.assertIsNone(llm.active())                  # key removed: not usable

    def test_environment_and_old_key_file(self):
        llm.OLD_KEY.write_text('old-key')
        self.assertEqual(llm.active()['provider'], 'anthropic')   # nothing chosen, old Anthropic key still works
        os.environ['GEMINI_API_KEY'] = 'g'
        llm.update(dict(provider='gemini', models={'gemini': 'gem'}))
        self.assertEqual(llm.active()['key'], 'g')

    def test_ollama_needs_no_key_and_urls_are_checked(self):
        llm.update(dict(provider='ollama'))
        self.assertEqual(llm.active()['base'], 'http://127.0.0.1:11434/v1')
        with self.assertRaises(ValueError): llm.update(dict(base_urls={'custom': 'file:///etc/passwd'}))
        with self.assertRaises(ValueError): llm.update(dict(provider='nope'))

    def test_visitor_rules(self):
        self.assertIsNone(llm.visitor(None)); self.assertIsNone(llm.visitor({'provider': 'openai'}))
        with self.assertRaises(llm.LLMError): llm.visitor({'provider': 'ollama', 'key': 'k', 'model': 'm'})
        with self.assertRaises(llm.LLMError): llm.visitor({'provider': 'openai', 'key': 'k'})   # model required
        self.assertEqual(llm.visitor({'provider': 'anthropic', 'key': 'k'})['model'], 'claude-sonnet-5-5')


class WireFormatTest(unittest.TestCase):
    def call(self, provider, reply, model='m', key='k'):
        p = llm.PROVIDERS[provider]
        cfg = dict(provider=provider, kind=p['kind'], label=p['label'], base=p['base'], model=model, key=key)
        seen = {}
        def fake(req, timeout): seen.update(url=req.full_url, headers={k.lower(): v for k, v in req.header_items()}, body=json.loads(req.data)); return Reply(reply)
        with mock.patch('urllib.request.urlopen', fake):
            out = llm.chat(cfg, 'SYS', [dict(role='user', content='hi'), dict(role='assistant', content='yo'), dict(role='user', content='q')])
        return out, seen

    def test_anthropic(self):
        out, s = self.call('anthropic', {'content': [{'type': 'text', 'text': 'A'}]})
        self.assertEqual(out, 'A'); self.assertTrue(s['url'].endswith('/v1/messages'))
        self.assertEqual(s['headers']['x-api-key'], 'k'); self.assertEqual(s['body']['system'], 'SYS')

    def test_openai_and_openrouter(self):
        out, s = self.call('openai', {'choices': [{'message': {'content': 'O'}}]})
        self.assertEqual(out, 'O'); self.assertEqual(s['url'], 'https://api.openai.com/v1/chat/completions')
        self.assertEqual(s['body']['messages'][0], {'role': 'system', 'content': 'SYS'}); self.assertIn('max_completion_tokens', s['body'])
        self.assertEqual(s['headers']['authorization'], 'Bearer k')
        out, s = self.call('openrouter', {'choices': [{'message': {'content': 'R'}}]})
        self.assertEqual(out, 'R'); self.assertIn('max_tokens', s['body']); self.assertTrue(s['url'].startswith('https://openrouter.ai'))

    def test_gemini(self):
        out, s = self.call('gemini', {'candidates': [{'content': {'parts': [{'text': 'G'}]}}]}, model='models x')
        self.assertEqual(out, 'G'); self.assertIn('/models/models%20x:generateContent', s['url'])
        self.assertEqual([c['role'] for c in s['body']['contents']], ['user', 'model', 'user'])
        self.assertEqual(s['headers']['x-goog-api-key'], 'k')

    def test_ollama_without_key(self):
        out, s = self.call('ollama', {'choices': [{'message': {'content': 'L'}}]}, key='')
        self.assertEqual(out, 'L'); self.assertNotIn('authorization', s['headers'])

    def test_errors_become_llm_errors(self):
        import urllib.error
        def down(req, timeout): raise urllib.error.URLError('refused')
        cfg = dict(provider='ollama', kind='openai', label='Ollama', base='http://127.0.0.1:1/v1', model='m', key='')
        with mock.patch('urllib.request.urlopen', down), self.assertRaises(llm.LLMError): llm.chat(cfg, 's', [dict(role='user', content='q')])


if __name__ == '__main__':
    unittest.main()
