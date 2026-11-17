# Hindi/Hinglish configuration comparison

Each experiment reuses its sample's exact decoded waveform. GPU figures are sampled total device memory, including other processes.

Verified-reference metrics are shown where available. Language preservation and translation errors still require human review.

| Source report | Model | Compute | Load seconds | Model-run sampled GPU MiB |
| --- | --- | --- | ---: | ---: |
| stt-medium-validation-candidate-20261116.json | C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\.models\faster-whisper-medium | int8_float16 | 6.254 | 1235 |
| stt-medium-validation-small-20261116.json | small | float16 | 3.760 | 787 |

## Corpus coverage and warm latency

Counts are unique recordings, not repeated passes. Missing references stay unscored. Warm latency excludes pass 1 for each sample/profile; OS file cache may be warm.

| Source report | Corpus language | Available / verified samples | Profile | Raw WER / CER | Normalized WER / CER | Warm median s |
| --- | --- | --- | --- | --- | --- | --- |
| stt-medium-validation-candidate-20261116.json | english | 0 / 0 | (none) | Unavailable | Unavailable | N/A |
| stt-medium-validation-candidate-20261116.json | hindi | 0 / 0 | (none) | Unavailable | Unavailable | N/A |
| stt-medium-validation-candidate-20261116.json | hinglish | 1 / 1 | baseline | 0.2105 / 0.1528 | 0.2105 / 0.1528 | 1.862 |
| stt-medium-validation-small-20261116.json | english | 0 / 0 | (none) | Unavailable | Unavailable | N/A |
| stt-medium-validation-small-20261116.json | hindi | 0 / 0 | (none) | Unavailable | Unavailable | N/A |
| stt-medium-validation-small-20261116.json | hinglish | 1 / 1 | baseline | 0.7895 / 0.5833 | 0.7895 / 0.5833 | 1.050 |

## Sample hinglish_technical

Path: C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\benchmark-recordings\hindi-investigation\hinglish_technical.npy

Decoded float32 SHA-256: `1adf58c7904830c398a02ca2145e9a4514a51968b4c0691e950f56cfc6add5c2`

| Source report | Model / compute | Profile / pass | VAD (threshold; silence ms; pad ms) | Language | Beam | Prompt | Raw transcript | Normalized transcript | Human reference | Raw WER / CER | Normalized WER / CER | Decode s | Discarded s | Sampled GPU MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stt-medium-validation-candidate-20261116.json | C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\.models\faster-whisper-medium / int8_float16 | baseline / 1 | on (0.5; 500; 300) | auto | 5 | (none) | आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.2105 / 0.1528 | 0.2105 / 0.1528 | 3.257 | 6.472 | 1235 |
| stt-medium-validation-candidate-20261116.json | C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\.models\faster-whisper-medium / int8_float16 | baseline / 2 | on (0.5; 500; 300) | auto | 5 | (none) | आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.2105 / 0.1528 | 0.2105 / 0.1528 | 1.834 | 6.472 | 1235 |
| stt-medium-validation-candidate-20261116.json | C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\.models\faster-whisper-medium / int8_float16 | baseline / 3 | on (0.5; 500; 300) | auto | 5 | (none) | आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.2105 / 0.1528 | 0.2105 / 0.1528 | 1.869 | 6.472 | 1235 |
| stt-medium-validation-candidate-20261116.json | C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\.models\faster-whisper-medium / int8_float16 | baseline / 4 | on (0.5; 500; 300) | auto | 5 | (none) | आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.2105 / 0.1528 | 0.2105 / 0.1528 | 1.854 | 6.472 | 1235 |
| stt-medium-validation-candidate-20261116.json | C:\Users\shiva\Downloads\VoiceOS_Project_Hardened\VoiceOS_Project\.models\faster-whisper-medium / int8_float16 | baseline / 5 | on (0.5; 500; 300) | auto | 5 | (none) | आज मैं voiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS प्रोजेक पे Python और Gemini इस्तमाल कर रहा हूँ और BBMS का काम अभी बाकी है। | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.2105 / 0.1528 | 0.2105 / 0.1528 | 1.918 | 6.472 | 1235 |
| stt-medium-validation-small-20261116.json | small / float16 | baseline / 1 | on (0.5; 500; 300) | auto | 5 | (none) | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.7895 / 0.5833 | 0.7895 / 0.5833 | 1.377 | 6.472 | 755 |
| stt-medium-validation-small-20261116.json | small / float16 | baseline / 2 | on (0.5; 500; 300) | auto | 5 | (none) | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.7895 / 0.5833 | 0.7895 / 0.5833 | 1.022 | 6.472 | 787 |
| stt-medium-validation-small-20261116.json | small / float16 | baseline / 3 | on (0.5; 500; 300) | auto | 5 | (none) | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.7895 / 0.5833 | 0.7895 / 0.5833 | 1.172 | 6.472 | 787 |
| stt-medium-validation-small-20261116.json | small / float16 | baseline / 4 | on (0.5; 500; 300) | auto | 5 | (none) | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.7895 / 0.5833 | 0.7895 / 0.5833 | 1.066 | 6.472 | 787 |
| stt-medium-validation-small-20261116.json | small / float16 | baseline / 5 | on (0.5; 500; 300) | auto | 5 | (none) | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज में वोईस हो इस प्रुजेक पे पाइटन और जेमिना इस तमाल कर रहा हों और भी भी एमस का काम अभी बाखी है | आज मैं VoiceOS project में Python और Gemini इस्तेमाल कर रहा हूँ, और DBMS का काम अभी बाकी है। | 0.7895 / 0.5833 | 0.7895 / 0.5833 | 1.034 | 6.472 | 787 |

VAD discard duration is signal duration, not proof of removed speech. Use the JSON interval coordinates and exported excerpts for human review.

