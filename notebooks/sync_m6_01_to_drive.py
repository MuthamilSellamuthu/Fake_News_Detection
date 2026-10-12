# ONE-FILE SYNC: write the memory-safe scripts/m6_01_encode_evidence.py to Drive.
#
# Why this exists: the notebook currently in Colab carries an OLD embedded
# payload, so every time its sync cell runs it rewrites the OLD script. This
# cell writes ONLY this one file and verifies it by SHA-256.
#
# AFTER running this, do NOT re-run the old notebook's sync cell - it would
# overwrite this file with the stale version again.
import base64, hashlib
from pathlib import Path

PROJECT = Path("/content/drive/MyDrive/Fake_News_Detection")
assert PROJECT.exists(), "Drive not mounted - run the mount cell first"

DEST = PROJECT / "scripts" / "m6_01_encode_evidence.py"
EXPECTED_SHA = "b5561ae0073260b80a43532f056149e2b64b5b682f31b7880bbc77d74c31d7d4"
EXPECTED_SIZE = 7634

B64 = (
    "IyEvdXNyL2Jpbi9lbnYgcHl0aG9uCiIiIk1vZHVsZSA2IFN0YWdlIDEg4oCUIGVuY29kZSB0aGUgZXZpZGVuY2UgcG9vbCB3aXRo"
    "IExhQlNFLiBHUFUuIFJlc3VtYWJsZS4KCkVuY29kZXMgb25seSB0aGUgZXZpZGVuY2UgcGVybWl0dGVkIGZvciB0aGUgcmVxdWVz"
    "dGVkIHBoYXNlLCBkZXJpdmVkIGZyb20KTW9kdWxlIDUncyBidWlsZF9pbmRleF9zY29wZSgpLiBFbmNvZGluZyB0aGUgd2hvbGUg"
    "cG9vbCBhbmQgZmlsdGVyaW5nIGxhdGVyCndvdWxkIHdvcmssIGJ1dCBzY29waW5nIGhlcmUgbWVhbnMgYW4gb3V0LW9mLXNjb3Bl"
    "IGl0ZW0gbmV2ZXIgZXZlbiBnZXRzIGEKdmVjdG9yIC0tIHRoZSBsZWFrYWdlIGd1YXJhbnRlZSBob2xkcyBieSBjb25zdHJ1Y3Rp"
    "b24sIG5vdCBieSBkaXNjaXBsaW5lLgoKU2hhcmRzIG9mIDEwLDAwMCwgYXRvbWljIHRlbXAtdGhlbi1yZW5hbWUsIG1hbmlmZXN0"
    "LWRyaXZlbiByZXN1bWUuCgogICAgcHl0aG9uIHNjcmlwdHMvbTZfMDFfZW5jb2RlX2V2aWRlbmNlLnB5IC0tZGF0YXNldCBmZXZl"
    "ciAtLXBoYXNlIHRlc3QKIiIiCmZyb20gX19mdXR1cmVfXyBpbXBvcnQgYW5ub3RhdGlvbnMKCmltcG9ydCBhcmdwYXJzZQppbXBv"
    "cnQgc3lzCmltcG9ydCB0aW1lCmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAoKc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihQYXRoKF9f"
    "ZmlsZV9fKS5yZXNvbHZlKCkucGFyZW50c1sxXSkpCgpmcm9tIHNjcmlwdHMubTZfY29tbW9uIGltcG9ydCAoICAjIG5vcWE6IEU0"
    "MDIKICAgIFBST0pFQ1RfUk9PVCwgYXNzZXJ0X21vZHVsZTVfdW50b3VjaGVkLCBiYW5uZXIsIGNvbmZpZ19oYXNoLCBkZXRlY3Rf"
    "ZGV2aWNlLAogICAgbG9hZF9jb25maWcsIGxvYWRfY29ycHVzLCBtb2R1bGU1X2ZpbmdlcnByaW50LCBzdGFnZV9kaXIsIHdyaXRl"
    "X21hbmlmZXN0KQpmcm9tIHNyYy5yZXRyaWV2YWwuZW5jb2RlcnMgaW1wb3J0ICggICMgbm9xYTogRTQwMgogICAgTG93TWVtb3J5"
    "U3RvcCwgU2hhcmRlZEVuY29kZXIsIGF2YWlsYWJsZV9tYiwgYnVpbGRfbGFic2VfZW5jb2RlX2ZuKQpmcm9tIHNyYy5yZXRyaWV2"
    "YWwuaW5kZXhfc2NvcGUgaW1wb3J0IGJ1aWxkX2V2aWRlbmNlX3Njb3BlICAjIG5vcWE6IEU0MDIKCgpkZWYgbWFpbigpIC0+IE5v"
    "bmU6CiAgICBhcCA9IGFyZ3BhcnNlLkFyZ3VtZW50UGFyc2VyKGRlc2NyaXB0aW9uPSJFbmNvZGUgc2NvcGVkIGV2aWRlbmNlIHdp"
    "dGggTGFCU0UuIikKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1kYXRhc2V0IiwgcmVxdWlyZWQ9VHJ1ZSwgY2hvaWNlcz1bImZldmVy"
    "IiwgInhfZmFjdCJdKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLXBoYXNlIiwgZGVmYXVsdD0idGVzdCIsIGNob2ljZXM9WyJ0cmFp"
    "biIsICJ2YWxpZGF0aW9uIiwgInRlc3QiXSkKICAgIGFwLmFkZF9hcmd1bWVudCgiLS1jb25maWciLCBkZWZhdWx0PU5vbmUpCiAg"
    "ICBhcC5hZGRfYXJndW1lbnQoIi0tcHJvamVjdC1yb290IiwgZGVmYXVsdD1Ob25lKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWRy"
    "eS1ydW4iLCBhY3Rpb249InN0b3JlX3RydWUiLAogICAgICAgICAgICAgICAgICAgIGhlbHA9IlJlcG9ydCB3aGF0IHdvdWxkIGJl"
    "IGVuY29kZWQ7IGRvd25sb2FkIG5vIG1vZGVsLiIpCiAgICBhcC5hZGRfYXJndW1lbnQoIi0tbW9kZWwiLCBkZWZhdWx0PU5vbmUs"
    "CiAgICAgICAgICAgICAgICAgICAgaGVscD0iT3ZlcnJpZGUgdGhlIGJpLWVuY29kZXIuIEEgc21hbGxlciBtb2RlbCAoZS5nLiAi"
    "CiAgICAgICAgICAgICAgICAgICAgICAgICAic2VudGVuY2UtdHJhbnNmb3JtZXJzL3BhcmFwaHJhc2UtbXVsdGlsaW5ndWFsLU1p"
    "bmlMTS1MMTItdjIpICIKICAgICAgICAgICAgICAgICAgICAgICAgICJjdXRzIFJBTSB+NHggaWYgTGFCU0UgaXMgYmVpbmcgT09N"
    "LWtpbGxlZC4iKQogICAgYXAuYWRkX2FyZ3VtZW50KCItLWJhdGNoLXNpemUiLCB0eXBlPWludCwgZGVmYXVsdD1Ob25lKQogICAg"
    "YXAuYWRkX2FyZ3VtZW50KCItLW1heC1zZXEtbGVuZ3RoIiwgdHlwZT1pbnQsIGRlZmF1bHQ9Tm9uZSkKICAgIGFwLmFkZF9hcmd1"
    "bWVudCgiLS1zaGFyZC1zaXplIiwgdHlwZT1pbnQsIGRlZmF1bHQ9Tm9uZSwKICAgICAgICAgICAgICAgICAgICBoZWxwPSJTbWFs"
    "bGVyIHNoYXJkcyBjaGVja3BvaW50IG1vcmUgb2Z0ZW4gYW5kIGNhcCBwZWFrIFJBTS4iKQogICAgYXAuYWRkX2FyZ3VtZW50KCIt"
    "LW1pbi1hdmFpbGFibGUtbWIiLCB0eXBlPWZsb2F0LCBkZWZhdWx0PTEyMDAuMCwKICAgICAgICAgICAgICAgICAgICBoZWxwPSJT"
    "dG9wIGNsZWFubHkgaWYgc3lzdGVtIFJBTSBkcm9wcyBiZWxvdyB0aGlzLCBpbnN0ZWFkIG9mICIKICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICJiZWluZyBPT00ta2lsbGVkIHdpdGggbm8gdHJhY2ViYWNrLiAwIGRpc2FibGVzLiIpCiAgICBhcC5hZGRfYXJndW1l"
    "bnQoIi0tbnVtLXRocmVhZHMiLCB0eXBlPWludCwgZGVmYXVsdD0yLAogICAgICAgICAgICAgICAgICAgIGhlbHA9InRvcmNoIENQ"
    "VSB0aHJlYWRzLiBFYWNoIGNhcnJpZXMgaXRzIG93biB3b3Jrc3BhY2UsIHNvIGFuICIKICAgICAgICAgICAgICAgICAgICAgICAg"
    "ICJ1bmJvdW5kZWQgY291bnQgbXVsdGlwbGllcyBwZWFrIFJBTSBvbiBhIHNtYWxsIGNvbnRhaW5lci4iKQogICAgYXJncyA9IGFw"
    "LnBhcnNlX2FyZ3MoKQoKICAgIHJvb3QgPSBQYXRoKGFyZ3MucHJvamVjdF9yb290KSBpZiBhcmdzLnByb2plY3Rfcm9vdCBlbHNl"
    "IFBST0pFQ1RfUk9PVAogICAgY2ZnID0gbG9hZF9jb25maWcoYXJncy5jb25maWcpCiAgICBiYW5uZXIoZiJTdGFnZSAxIGVuY29k"
    "ZSBldmlkZW5jZSBbe2FyZ3MuZGF0YXNldH0ve2FyZ3MucGhhc2V9XSIsIGNmZykKICAgIGJlZm9yZSA9IG1vZHVsZTVfZmluZ2Vy"
    "cHJpbnQocm9vdCkKCiAgICBjb3JwdXMgPSBsb2FkX2NvcnB1cyhjZmcsIGFyZ3MuZGF0YXNldCwgcm9vdD1yb290KQogICAgc2Nv"
    "cGUgPSBidWlsZF9ldmlkZW5jZV9zY29wZShjb3JwdXNbImNsYWltcyJdLCBjb3JwdXNbInNwbGl0cyJdLCBhcmdzLnBoYXNlLAog"
    "ICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBjZmdbImxlYWthZ2UiXVsiaW5kZXhfc2NvcGUiXSkKICAgIGV2X2lkcyA9"
    "IHNvcnRlZChzY29wZS5ldmlkZW5jZV9pZHMpICAgICAgICAgICAgICAgICAjIHNvcnRlZCA9PiBkZXRlcm1pbmlzdGljIHNoYXJk"
    "cwogICAgZXZfdGV4dHMgPSBjb3JwdXNbImV2aWRlbmNlX3RleHRzIl0KICAgICMgVmlld3MsIG5vdCBhIHNlY29uZCBmdWxsIGNv"
    "cHkgb2YgdGhlIGNvcnB1cyB0ZXh0IChyZXF1aXJlbWVudDogbmV2ZXIgaG9sZAogICAgIyB0aGUgd2hvbGUgcG9vbCB0d2ljZSku"
    "IHRleHRfZm4gbWF0ZXJpYWxpc2VzIG9uZSBzaGFyZCBhdCBhIHRpbWUuCiAgICB0ZXh0cyA9IFtldl90ZXh0cy5nZXQoZSwgIiIp"
    "IGZvciBlIGluIGV2X2lkc10KCiAgICBkZWYgdGV4dF9mbihsbzogaW50LCBoaTogaW50KSAtPiBsaXN0W3N0cl06CiAgICAgICAg"
    "cmV0dXJuIFtldl90ZXh0cy5nZXQoZSwgIiIpIGZvciBlIGluIGV2X2lkc1tsbzpoaV1dCgogICAgcHJpbnQoZiIgIGNsYWltcyBp"
    "biBzY29wZSAgIDoge3Njb3BlLm5fY2xhaW1zX2luX3Njb3BlOix9IikKICAgIHByaW50KGYiICBldmlkZW5jZSBpbiBzY29wZSA6"
    "IHtsZW4oZXZfaWRzKTosfSIpCiAgICBwcmludChmIiAgZXZpZGVuY2UgZXhjbHVkZWQgOiB7c2NvcGUubl9ldmlkZW5jZV9leGNs"
    "dWRlZDosfSAob3V0LW9mLXNjb3BlIG9ubHkpIikKCiAgICBvdXQgPSBzdGFnZV9kaXIoY2ZnLCAiZW5jb2RlX2V2aWRlbmNlIiwg"
    "cm9vdCkgLyBmInthcmdzLmRhdGFzZXR9X3thcmdzLnBoYXNlfSIKICAgIG91dC5ta2RpcihwYXJlbnRzPVRydWUsIGV4aXN0X29r"
    "PVRydWUpCgogICAgaWYgYXJncy5kcnlfcnVuOgogICAgICAgIHdyaXRlX21hbmlmZXN0KG91dCwgeyJzdGFnZSI6ICJlbmNvZGVf"
    "ZXZpZGVuY2UiLCAiZGF0YXNldCI6IGFyZ3MuZGF0YXNldCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAicGhhc2UiOiBh"
    "cmdzLnBoYXNlLCAic3RhdHVzIjogImRyeV9ydW4iLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICJuX2V2aWRlbmNlIjog"
    "bGVuKGV2X2lkcyksICJzY29wZSI6IHNjb3BlLnN0YXRzfSkKICAgICAgICBwcmludCgiICBEUlkgUlVOOiBubyBtb2RlbCBsb2Fk"
    "ZWQsIG5vIHZlY3RvcnMgd3JpdHRlbi4iKQogICAgICAgIGFzc2VydF9tb2R1bGU1X3VudG91Y2hlZChiZWZvcmUsIHJvb3QpCiAg"
    "ICAgICAgcmV0dXJuCgogICAgciA9IGNmZ1sicmV0cmlldmFsIl0KICAgIGRldmljZSA9IGRldGVjdF9kZXZpY2UoKQogICAgIyBD"
    "UFUtc2FmZSBkZWZhdWx0cy4gVGhlIGNvbmZpZyB2YWx1ZXMgKGJhdGNoIDY0IC8gc2VxIDI1NiAvIHNoYXJkIDEwMDAwKSBhcmUK"
    "ICAgICMgc2l6ZWQgZm9yIGEgR1BVOyBvbiBDUFUgdGhhdCBzaGFyZCBzaXplIG1lYW5zIHRoZSB3aG9sZSA2LDk2OC1pdGVtIHBv"
    "b2wgZ29lcwogICAgIyB0aHJvdWdoIG9uZSBlbmNvZGUoKSBjYWxsLCBhbmQgdGhlIHBlYWsgaXMgd2hhdCBnZXRzIHRoZSBwcm9j"
    "ZXNzIE9PTS1raWxsZWQuCiAgICBjcHUgPSBkZXZpY2UgPT0gImNwdSIKICAgIG1vZGVsX25hbWUgPSBhcmdzLm1vZGVsIG9yIHJb"
    "ImJpX2VuY29kZXIiXQogICAgYmF0Y2ggPSBpbnQoYXJncy5iYXRjaF9zaXplIG9yICg4IGlmIGNwdSBlbHNlIHJbImVuY29kZV9i"
    "YXRjaF9zaXplIl0pKQogICAgc2VxID0gaW50KGFyZ3MubWF4X3NlcV9sZW5ndGggb3IgKDEyOCBpZiBjcHUgZWxzZSByWyJtYXhf"
    "c2VxX2xlbmd0aCJdKSkKICAgIHNoYXJkID0gaW50KGFyZ3Muc2hhcmRfc2l6ZSBvciAoNTAwIGlmIGNwdSBlbHNlIHJbInNoYXJk"
    "X3NpemUiXSkpCiAgICBzZXR0aW5ncyA9IHsibW9kZWwiOiBtb2RlbF9uYW1lLCAiYmF0Y2hfc2l6ZSI6IGJhdGNoLCAibWF4X3Nl"
    "cV9sZW5ndGgiOiBzZXF9CgogICAgcHJpbnQoZiIgIG1vZGVsICAgICAgOiB7bW9kZWxfbmFtZX0iKQogICAgcHJpbnQoZiIgIGJh"
    "dGNoPXtiYXRjaH0gIHNlcT17c2VxfSAgc2hhcmQ9e3NoYXJkfSAgdGhyZWFkcz17YXJncy5udW1fdGhyZWFkc30iCiAgICAgICAg"
    "ICArICgiICAgW0NQVS1zYWZlIGRlZmF1bHRzXSIgaWYgY3B1IGFuZCBub3QgKGFyZ3MuYmF0Y2hfc2l6ZSBvciBhcmdzLm1heF9z"
    "ZXFfbGVuZ3RoCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgb3IgYXJncy5zaGFy"
    "ZF9zaXplKSBlbHNlICIiKSkKICAgIHByaW50KGYiICBSQU0gZmxvb3IgIDoge2FyZ3MubWluX2F2YWlsYWJsZV9tYjouMGZ9IE1C"
    "ICIKICAgICAgICAgIGYiKGF2YWlsYWJsZSBub3cge2F2YWlsYWJsZV9tYigpOi4wZn0gTUIpIikKICAgIG5fc2hhcmRzID0gKGxl"
    "bihldl9pZHMpICsgc2hhcmQgLSAxKSAvLyBzaGFyZAogICAgcHJpbnQoZiIgIHNoYXJkcyAgICAgOiB7bl9zaGFyZHN9ICAoe2xl"
    "bihldl9pZHMpOix9IGl0ZW1zKSIpCgogICAgZW5jID0gU2hhcmRlZEVuY29kZXIob3V0LCBzaGFyZF9zaXplPXNoYXJkLAogICAg"
    "ICAgICAgICAgICAgICAgICAgICAgZW5jb2RlX2ZuPWJ1aWxkX2xhYnNlX2VuY29kZV9mbigKICAgICAgICAgICAgICAgICAgICAg"
    "ICAgICAgICBtb2RlbF9uYW1lLCBiYXRjaF9zaXplPWJhdGNoLCBtYXhfc2VxX2xlbmd0aD1zZXEsCiAgICAgICAgICAgICAgICAg"
    "ICAgICAgICAgICAgZGV2aWNlPWRldmljZSwgbnVtX3RocmVhZHM9aW50KGFyZ3MubnVtX3RocmVhZHMpLAogICAgICAgICAgICAg"
    "ICAgICAgICAgICAgICAgIGxvZz1wcmludCksCiAgICAgICAgICAgICAgICAgICAgICAgICBtb2RlbF9uYW1lPW1vZGVsX25hbWUp"
    "CgogICAgdDAgPSB0aW1lLnBlcmZfY291bnRlcigpCiAgICByYW1fbG8gPSBhdmFpbGFibGVfbWIoKQogICAgc3RvcHBlZCA9IE5v"
    "bmUKICAgIHRyeToKICAgICAgICBtYW4gPSBlbmMuZW5jb2RlKHRleHRzLCBpZHM9ZXZfaWRzLCBtaW5fYXZhaWxhYmxlX21iPWZs"
    "b2F0KGFyZ3MubWluX2F2YWlsYWJsZV9tYiksCiAgICAgICAgICAgICAgICAgICAgICAgICBzZXR0aW5ncz1zZXR0aW5ncywgdGV4"
    "dF9mbj10ZXh0X2ZuKQogICAgZXhjZXB0IExvd01lbW9yeVN0b3AgYXMgZXhjOgogICAgICAgIHN0b3BwZWQgPSBzdHIoZXhjKQog"
    "ICAgICAgIG1hbiA9IGVuYy5sb2FkX21hbmlmZXN0KCkKCiAgICBtYW4udXBkYXRlKHsic3RhZ2UiOiAiZW5jb2RlX2V2aWRlbmNl"
    "IiwgImRhdGFzZXQiOiBhcmdzLmRhdGFzZXQsICJwaGFzZSI6IGFyZ3MucGhhc2UsCiAgICAgICAgICAgICAgICAic2NvcGUiOiBz"
    "Y29wZS5zdGF0cywgInNldHRpbmdzIjogc2V0dGluZ3MsICJkZXZpY2UiOiBkZXZpY2UsCiAgICAgICAgICAgICAgICAiY29uZmln"
    "X2hhc2giOiBjb25maWdfaGFzaChjZmcsIFsicmV0cmlldmFsIiwgImxlYWthZ2UiXSl9KQogICAgd3JpdGVfbWFuaWZlc3Qob3V0"
    "LCBtYW4pCiAgICBhc3NlcnRfbW9kdWxlNV91bnRvdWNoZWQoYmVmb3JlLCByb290KQoKICAgIGRvbmUgPSBzb3J0ZWQobWFuLmdl"
    "dCgiY29tcGxldGVkX3NoYXJkcyIsIFtdKSkKICAgIHJlbWFpbmluZyA9IFtpIGZvciBpIGluIHJhbmdlKG5fc2hhcmRzKSBpZiBp"
    "IG5vdCBpbiBzZXQoZG9uZSldCiAgICB3cml0dGVuID0gc3VtKDEgZm9yIGkgaW4gcmFuZ2Uobl9zaGFyZHMpIGlmIChvdXQgLyBm"
    "InNoYXJkX3tpOjA1ZH0ubnB5IikuaXNfZmlsZSgpKQogICAgcmFtX25vdyA9IGF2YWlsYWJsZV9tYigpCiAgICBwcmludCgiXG4i"
    "ICsgIi0iICogNjIpCiAgICBwcmludCgiU1RBR0UgMWEgU1VNTUFSWSIpCiAgICBwcmludCgiLSIgKiA2MikKICAgIHByaW50KGYi"
    "ICBjb21wbGV0ZWQgc2hhcmRzICAgOiB7bGVuKGRvbmUpfS97bl9zaGFyZHN9IikKICAgIHByaW50KGYiICByZW1haW5pbmcgc2hh"
    "cmRzICAgOiB7bGVuKHJlbWFpbmluZyl9IgogICAgICAgICAgKyAoZiIgIChuZXh0OiB7cmVtYWluaW5nWzBdfSkiIGlmIHJlbWFp"
    "bmluZyBlbHNlICIiKSkKICAgIHByaW50KGYiICBlbWJlZGRpbmdzIHdyaXR0ZW4gOiB7d3JpdHRlbn0gc2hhcmQgZmlsZShzKSIp"
    "CiAgICBwcmludChmIiAgZWxhcHNlZCAgICAgICAgICAgIDoge3RpbWUucGVyZl9jb3VudGVyKCkgLSB0MDouMWZ9cyIpCiAgICBw"
    "cmludChmIiAgUkFNIGF2YWlsYWJsZSAgICAgIDogc3RhcnQge3JhbV9sbzouMGZ9IE1CIC0+IG5vdyB7cmFtX25vdzouMGZ9IE1C"
    "IgogICAgICAgICAgZiIgIChsb3dlc3Qgc2VlbiB7bWFuLmdldCgnbWluX2F2YWlsYWJsZV9tYl9vYnNlcnZlZCcpIG9yIHJhbV9u"
    "b3c6LjBmfSBNQikiKQogICAgcHJpbnQoZiIgIHN0YXR1cyAgICAgICAgICAgICA6IHttYW4uZ2V0KCdzdGF0dXMnKX0iKQogICAg"
    "cHJpbnQoZiIgIHJlc3VtZS1zYWZlICAgICAgICA6IFlFUyAtIHJlLXJ1biB0aGUgaWRlbnRpY2FsIGNvbW1hbmQ7ICIKICAgICAg"
    "ICAgIGYiY29tcGxldGVkIHNoYXJkcyBhcmUgc2tpcHBlZCIpCiAgICBwcmludChmIiAgb3V0cHV0ICAgICAgICAgICAgIDoge291"
    "dH0iKQogICAgcHJpbnQoIiAgTW9kdWxlIDUgdW50b3VjaGVkIDogVHJ1ZSIpCiAgICBpZiBzdG9wcGVkOgogICAgICAgIHByaW50"
    "KCJcbiAgU1RPUFBFRCBFQVJMWSAoY2xlYW4sIG5vdCBhIGNyYXNoKToiKQogICAgICAgIHByaW50KGYiICB7c3RvcHBlZH0iKQog"
    "ICAgICAgIHN5cy5leGl0KDIpCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIG1haW4oKQo="
)

old = DEST.stat().st_size if DEST.exists() else 0
DEST.parent.mkdir(parents=True, exist_ok=True)
DEST.write_bytes(base64.b64decode(B64))

got = hashlib.sha256(DEST.read_bytes()).hexdigest()
size = DEST.stat().st_size
print("path :", DEST)
print("size :", old, "->", size, "bytes")
print("sha256:", got)
assert got == EXPECTED_SHA and size == EXPECTED_SIZE, "WRITE VERIFY FAILED"

# clear stale bytecode so Python cannot import the old module
import shutil
pc = PROJECT / "scripts" / "__pycache__"
if pc.exists():
    shutil.rmtree(pc, ignore_errors=True)

text = DEST.read_text(encoding="utf-8")
print()
print("argument check:")
for a in ["--batch-size", "--max-seq-length", "--shard-size",
          "--num-threads", "--min-available-mb"]:
    print("  ", "FOUND  " if a in text else "MISSING", a)
print()
print("implementation check:")
for k in ["torch.inference_mode", "gc.collect", "set_num_threads",
          "LowMemoryStop", "sys.exit(2)"]:
    print("  ", "FOUND  " if k in text else "MISSING", k)
assert all(a in text for a in ["--batch-size", "--max-seq-length", "--shard-size",
                               "--num-threads", "--min-available-mb"])
print()
print("Drive copy is now the memory-safe version. Do not run Stage 1 yet.")
