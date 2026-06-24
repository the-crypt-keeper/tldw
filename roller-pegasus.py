#!/usr/bin/env python3
# Rolling video summarization backend using TwelveLabs Pegasus.
#
# Unlike the other roller-*.py backends, this one does NOT summarize the
# diarized transcript - it summarizes the video itself. Pegasus is a
# video-understanding model, so it sees what is on screen, not just what is
# said. To handle arbitrarily long videos despite the per-request 1 hour
# analyze limit, the video is analyzed in rolling time-windows, each window
# carrying forward a running context just like the other rollers.
#
# Setup:
#   pip install twelvelabs fire jinja2
#   export TWELVELABS_API_KEY=...   # free tier: https://twelvelabs.io
#
# Usage mirrors the other rollers - point it at a prefix that has a
# <prefix>.info.json (produced by diarize.py). It writes <prefix>.summary.json
# in the same line-per-section format the other rollers use. The source video
# is fetched server-side from the info.json webpage_url, so it must be a
# publicly accessible URL (or pass --video_url=).
import json
import os

instruction = """Continue the rolling summary of the video "{{title}}".
Consider the running context below when summarizing this segment, which covers
{{ start|round|int }}s to {{ end|round|int }}s of the video.

### Context: {{ context }}

Respond ONLY with a JSON object with 2 keys in the following format:
{
 "Summary": "A detailed, point-by-point summary of what happens in this segment of the video. Describe what is shown and what is said. Write at least three sentences and no more than six sentences. ALWAYS maintain third person.",
 "Next-Context": "An updated context for the next segment. List the current topics and any people identified so far."
}
"""


def main(prefix: str,
         model_name: str = "pegasus1.5",
         window: int = 600,
         max_tokens: int = 2048,
         video_url: str = ""):
    """Summarize a video with TwelveLabs Pegasus in rolling time-windows.

    prefix:     path prefix; reads <prefix>.info.json, writes <prefix>.summary.json
    model_name: pegasus1.5 (default; required for time-window analyze)
    window:     seconds per rolling analyze window (must be <= 3600)
    video_url:  override the source URL (defaults to info.json webpage_url)
    """
    from jinja2 import Template
    from twelvelabs import TwelveLabs
    from twelvelabs.types.video_context import VideoContext_Url

    if window > 3600:
        raise ValueError("window must be <= 3600s (Pegasus analyze caps at 1 hour per request)")

    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError("TWELVELABS_API_KEY is not set (free key: https://twelvelabs.io)")

    info = json.load(open(prefix + '.info.json'))
    title = info['title']
    url = video_url or info.get('webpage_url') or info.get('original_url')
    if not url:
        raise RuntimeError("no video URL: pass video_url= or ensure info.json has webpage_url")
    duration = info.get('duration')

    client = TwelveLabs(api_key=api_key)
    video = VideoContext_Url(url=url)

    the_template = Template(instruction)
    context = f'Title: "{title}"\nTopics: [ "UNKNOWN" ]'

    # Without a known duration we cannot window, so do a single full-video pass.
    if not duration:
        windows = [(0.0, None)]
    else:
        windows = [(float(s), float(min(s + window, duration)))
                   for s in range(0, int(duration), window)]

    f = open(prefix + '.summary.json', 'w')
    for idx, (start, end) in enumerate(windows):
        print(f"{idx}: {start}s -> {end}s")
        prompt = the_template.render(title=title, context=context, start=start, end=end or 0)

        res = client.analyze(
            model_name=model_name,
            video=video,
            prompt=prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            start_time=start,
            end_time=end,
        )
        answer = res.data or ''
        if not answer.endswith('}'):
            answer += '}'

        try:
            parsed = json.loads(answer, strict=False)
        except Exception as e:
            print(answer)
            print('Error parsing response: ', str(e))
            parsed = {}

        summary = parsed.get('Summary', '')
        new_context = str(parsed.get('Next-Context', ''))

        if summary == '' or new_context == '':
            print('extraction failed:', new_context, summary)
            exit(1)

        section = {
            'start': start,
            'end': end if end is not None else (duration or start),
            'summary': summary,
            'speakers': '{}',  # Pegasus summarizes video, not diarized speakers
            'context': new_context,
        }
        print('>> ', new_context)
        print(summary)
        print()

        f.write(json.dumps(section) + '\n')
        f.flush()

        context = new_context


if __name__ == "__main__":
    import fire
    fire.Fire(main)
