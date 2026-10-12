# =====================================================================
# MODULE 6 - FIX THE FEVER DENSE-RETRIEVAL OOM (exit -9)
#
# Self-contained: the corrected faiss_index.py is embedded below as base64,
# so this does NOT depend on any Drive file appearing, on which account owns
# what, or on FUSE cache propagation.
#
# STEP 1 diagnoses where things actually are.
# STEP 2 writes the fixed file and verifies it by SHA-256.
# =====================================================================
import base64, hashlib, shutil
from pathlib import Path

print("STEP 1 - DIAGNOSTIC")
print("-" * 62)
mounted = Path("/content/drive").exists()
print("  /content/drive exists      :", mounted)
if not mounted:
    raise SystemExit(
        "Drive is NOT mounted. Run this first, then re-run this cell:\n"
        "    from google.colab import drive; drive.mount('/content/drive')")

PROJECT = Path("/content/drive/MyDrive/Fake_News_Detection")
print("  project root exists        :", PROJECT.exists())
if not PROJECT.exists():
    print()
    print("  Directories in MyDrive:")
    for p in sorted(Path("/content/drive/MyDrive").iterdir()):
        if p.is_dir():
            print("     ", p.name)
    raise SystemExit(
        "Project root not found under MyDrive.\n"
        "  The project is OWNED BY s.muthamil05@gmail.com. If this Colab session\n"
        "  is signed in as a different account, the folder appears under\n"
        "  'Shared with me' and is NOT visible at MyDrive/... unless you first\n"
        "  right-click it in Drive and choose 'Add shortcut to Drive'.")

RET = PROJECT / "src" / "retrieval"
print("  src/ exists                :", (PROJECT / "src").exists())
print("  src/retrieval/ exists      :", RET.exists())
if RET.exists():
    print("  contents of src/retrieval/ :")
    for p in sorted(RET.iterdir()):
        size = str(p.stat().st_size) + " bytes" if p.is_file() else "<dir>"
        print("      " + p.name.ljust(30) + size)
else:
    print("  !! src/retrieval/ missing. Searching project for faiss_index*.py ...")
    for hit in PROJECT.rglob("faiss_index*.py"):
        print("      found:", hit.relative_to(PROJECT), hit.stat().st_size, "bytes")

print()
print("STEP 2 - WRITE THE CORRECTED FILE")
print("-" * 62)
EXPECTED_SHA = "8a2c74b82ab3da48be14a31f170b282cd5bf7d4152e129896f71b9c3f96eda21"
EXPECTED_SIZE = 7288

B64 = (
    "IiIiTW9kdWxlIDYg4oCUIEZBSVNTIGZsYXQgaW5uZXItcHJvZHVjdCBpbmRleCBvdmVyIG5vcm1hbGlzZWQgZW1iZWRkaW"
    "5ncy4KCkluZGV4RmxhdElQIG9uIEwyLW5vcm1hbGlzZWQgdmVjdG9ycyBJUyBleGFjdCBjb3NpbmUgc2ltaWxhcml0eSwg"
    "YW5kICJmbGF0IgptZWFucyBleGhhdXN0aXZlOiBubyBhcHByb3hpbWF0aW9uLCBubyByZWNhbGwgbG9zcywgbm8gdHJhaW"
    "5pbmcgc3RlcCwgYW5kCnRoZXJlZm9yZSBub3RoaW5nIHN0b2NoYXN0aWMgdG8gcmVwcm9kdWNlLiBBdCB0aGlzIGNvcnB1"
    "cyBzaXplICg2NywzODggRkVWRVIKYW5kIDYsOTY4IFgtRkFDVCBldmlkZW5jZSBpdGVtcykgYW4gYXBwcm94aW1hdGUgaW"
    "5kZXggd291bGQgYnV5IGxpdHRsZSBhbmQKY29zdCBleGFjdG5lc3MsIHNvIHRoZSBmbGF0IGluZGV4IGlzIHRoZSByaWdo"
    "dCB0cmFkZSBhbmQga2VlcHMgdGhlIHJldHJpZXZhbApudW1iZXJzIGF1ZGl0YWJsZS4KCkRldGVybWluaXNtOiB2ZWN0b3"
    "JzIGFyZSBpbmRleGVkIGluIGEgZml4ZWQsIGNhbGxlci1zdXBwbGllZCBpZCBvcmRlciwgYW5kCnRpZXMgaW4gdGhlIHNj"
    "b3JlIGFyZSBicm9rZW4gYnkgdGhhdCBvcmRlci4gYHNlYXJjaCgpYCBpcyB0aGVyZWZvcmUKcmVwcm9kdWNpYmxlIGFjcm"
    "9zcyBydW5zIGFuZCBtYWNoaW5lcy4KCkEgcHVyZS1OdW1QeSBmYWxsYmFjayBpcyB1c2VkIHdoZW4gZmFpc3MgaXMgdW5h"
    "dmFpbGFibGUsIHNvIHRlc3RzIGFuZCBDUFUKc3RhZ2VzIHJ1biBhbnl3aGVyZS4gYHRlc3RzL3Rlc3RfcmV0cmlldmFsLn"
    "B5YCBhc3NlcnRzIHRoZSB0d28gYWdyZWUuCiIiIgoKZnJvbSBfX2Z1dHVyZV9fIGltcG9ydCBhbm5vdGF0aW9ucwoKaW1w"
    "b3J0IGpzb24KZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCmZyb20gdHlwaW5nIGltcG9ydCBBbnksIFNlcXVlbmNlCgppbX"
    "BvcnQgbnVtcHkgYXMgbnAKCnRyeTogICMgcHJhZ21hOiBubyBjb3ZlciAtIGF2YWlsYWJpbGl0eSBkaWZmZXJzIGJ5IGVu"
    "dmlyb25tZW50CiAgICBpbXBvcnQgZmFpc3MgICMgdHlwZTogaWdub3JlCiAgICBfSEFTX0ZBSVNTID0gVHJ1ZQpleGNlcH"
    "QgRXhjZXB0aW9uOiAgIyBub3FhOiBCTEUwMDEKICAgIGZhaXNzID0gTm9uZSAgIyB0eXBlOiBpZ25vcmUKICAgIF9IQVNf"
    "RkFJU1MgPSBGYWxzZQoKCmRlZiBmYWlzc19hdmFpbGFibGUoKSAtPiBib29sOgogICAgcmV0dXJuIF9IQVNfRkFJU1MKCg"
    "pkZWYgbDJfbm9ybWFsaXplKHg6IG5wLm5kYXJyYXkpIC0+IG5wLm5kYXJyYXk6CiAgICAiIiJSb3ctd2lzZSBMMiBub3Jt"
    "YWxpc2F0aW9uLiBaZXJvIHJvd3Mgc3RheSB6ZXJvIHJhdGhlciB0aGFuIGJlY29taW5nIE5hTi4iIiIKICAgIHggPSBucC"
    "5hc2FycmF5KHgsIGR0eXBlPW5wLmZsb2F0MzIpCiAgICBub3JtcyA9IG5wLmxpbmFsZy5ub3JtKHgsIGF4aXM9MSwga2Vl"
    "cGRpbXM9VHJ1ZSkKICAgIHJldHVybiB4IC8gbnAubWF4aW11bShub3JtcywgMWUtMTIpCgoKY2xhc3MgRXZpZGVuY2VJbm"
    "RleDoKICAgICIiIkV4YWN0IGlubmVyLXByb2R1Y3QgaW5kZXggd2l0aCBhbiBleHBsaWNpdCBpZCBvcmRlcmluZy4iIiIK"
    "CiAgICBkZWYgX19pbml0X18oc2VsZiwgZXZpZGVuY2VfaWRzOiBTZXF1ZW5jZVtzdHJdLCB2ZWN0b3JzOiBucC5uZGFycm"
    "F5LAogICAgICAgICAgICAgICAgIG5vcm1hbGl6ZTogYm9vbCA9IFRydWUsIHVzZV9mYWlzczogYm9vbCB8IE5vbmUgPSBO"
    "b25lLAogICAgICAgICAgICAgICAgIHZlY3RvcnNfcHJlbm9ybWFsaXplZDogYm9vbCA9IEZhbHNlKSAtPiBOb25lOgogIC"
    "AgICAgICIiImBub3JtYWxpemVgIGdvdmVybnMgQk9USCBzdG9yZWQgdmVjdG9ycyBhbmQgaW5jb21pbmcgcXVlcmllcy4K"
    "CiAgICAgICAgYHZlY3RvcnNfcHJlbm9ybWFsaXplZGAgc2F5cyB0aGUgdmVjdG9ycyBhcmUgYWxyZWFkeSB1bml0LWxlbm"
    "d0aCAoYXMKICAgICAgICB3aGVuIHJlbG9hZGluZyBmcm9tIGRpc2spIHNvIHRoZXkgYXJlIG5vdCBub3JtYWxpc2VkIHR3"
    "aWNlIC0tIGJ1dAogICAgICAgIHF1ZXJpZXMgc3RpbGwgYXJlLiBDb25mbGF0aW5nIHRoZXNlIHR3byBpcyBhIHN1YnRsZS"
    "BidWc6IGFuIGluZGV4CiAgICAgICAgcmVsb2FkZWQgd2l0aCBgbm9ybWFsaXplPUZhbHNlYCB3b3VsZCBzdG9wIG5vcm1h"
    "bGlzaW5nIHF1ZXJpZXMgYW5kCiAgICAgICAgc2lsZW50bHkgcmV0dXJuIGlubmVyIHByb2R1Y3RzIGluc3RlYWQgb2YgY2"
    "9zaW5lcy4KICAgICAgICAiIiIKICAgICAgICBpZiBsZW4oZXZpZGVuY2VfaWRzKSAhPSB2ZWN0b3JzLnNoYXBlWzBdOgog"
    "ICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKGYie2xlbihldmlkZW5jZV9pZHMpfSBpZHMgYnV0IHt2ZWN0b3JzLnNoYX"
    "BlWzBdfSB2ZWN0b3JzIikKICAgICAgICBpZiBsZW4oc2V0KGV2aWRlbmNlX2lkcykpICE9IGxlbihldmlkZW5jZV9pZHMp"
    "OgogICAgICAgICAgICByYWlzZSBWYWx1ZUVycm9yKCJkdXBsaWNhdGUgZXZpZGVuY2UgaWRzOiBpbmRleCBvcmRlciB3b3"
    "VsZCBiZSBhbWJpZ3VvdXMiKQogICAgICAgIHNlbGYuZXZpZGVuY2VfaWRzID0gbGlzdChldmlkZW5jZV9pZHMpCiAgICAg"
    "ICAgc2VsZi5kaW0gPSBpbnQodmVjdG9ycy5zaGFwZVsxXSkKICAgICAgICBzZWxmLm5vcm1hbGl6ZWQgPSBub3JtYWxpem"
    "UKICAgICAgICB2ZWNzID0gKG5wLmFzYXJyYXkodmVjdG9ycywgZHR5cGU9bnAuZmxvYXQzMikKICAgICAgICAgICAgICAg"
    "IGlmICh2ZWN0b3JzX3ByZW5vcm1hbGl6ZWQgb3Igbm90IG5vcm1hbGl6ZSkgZWxzZSBsMl9ub3JtYWxpemUodmVjdG9ycy"
    "kpCiAgICAgICAgc2VsZi52ZWN0b3JzID0gbnAuYXNjb250aWd1b3VzYXJyYXkodmVjcywgZHR5cGU9bnAuZmxvYXQzMikK"
    "ICAgICAgICBzZWxmLnVzZV9mYWlzcyA9IF9IQVNfRkFJU1MgaWYgdXNlX2ZhaXNzIGlzIE5vbmUgZWxzZSAodXNlX2ZhaX"
    "NzIGFuZCBfSEFTX0ZBSVNTKQogICAgICAgIHNlbGYuX2luZGV4ID0gTm9uZQogICAgICAgIGlmIHNlbGYudXNlX2ZhaXNz"
    "OgogICAgICAgICAgICBzZWxmLl9pbmRleCA9IGZhaXNzLkluZGV4RmxhdElQKHNlbGYuZGltKQogICAgICAgICAgICBzZW"
    "xmLl9pbmRleC5hZGQoc2VsZi52ZWN0b3JzKQoKICAgIGRlZiBfX2xlbl9fKHNlbGYpIC0+IGludDoKICAgICAgICByZXR1"
    "cm4gbGVuKHNlbGYuZXZpZGVuY2VfaWRzKQoKICAgIGRlZiBzZWFyY2goc2VsZiwgcXVlcmllczogbnAubmRhcnJheSwgdG"
    "9wX2s6IGludCwKICAgICAgICAgICAgICAgYmF0Y2hfc2l6ZTogaW50ID0gMTI4KSAtPiB0dXBsZVtsaXN0W2xpc3Rbc3Ry"
    "XV0sIG5wLm5kYXJyYXldOgogICAgICAgICIiIlRvcC1rIGV2aWRlbmNlIGlkcyBhbmQgc2NvcmVzIHBlciBxdWVyeSByb3"
    "cuCgogICAgICAgIGB0b3Bfa2AgaXMgY2xhbXBlZCB0byB0aGUgaW5kZXggc2l6ZSBzbyBhIHNtYWxsIHNjb3BlIGNhbm5v"
    "dCByYWlzZS4KCiAgICAgICAgUVVFUklFUyBBUkUgUFJPQ0VTU0VEIElOIEJBVENIRVMuIFRoZSBOdW1QeSBmYWxsYmFjay"
    "BtYXRlcmlhbGlzZXMgYQogICAgICAgIChuX3F1ZXJpZXMgeCBuX2V2aWRlbmNlKSBzaW1pbGFyaXR5IG1hdHJpeDsgYXQg"
    "RkVWRVIgc2NhbGUgdGhhdCBpcwogICAgICAgIDE1LDkxNiB4IDY3LDM4OCA9IDEuMDdlOSBmbG9hdDMyID0gNC4zIEdCLC"
    "BhbmQgdGhlIHN1YnNlcXVlbnQgYXJnc29ydAogICAgICAgIGFsbG9jYXRlcyBhIGZ1cnRoZXIgOC42IEdCIG9mIGludDY0"
    "LiBUb2dldGhlciB0aGV5IGV4Y2VlZCBhIHN0YW5kYXJkCiAgICAgICAgQ29sYWIgcnVudGltZSBhbmQgdGhlIHByb2Nlc3"
    "MgaXMgU0lHS0lMTGVkIChleGl0IC05KSB3aXRoIG5vIHRyYWNlYmFjay4KICAgICAgICBCYXRjaGluZyBib3VuZHMgdGhl"
    "IHBlYWsgYXQgcm91Z2hseQogICAgICAgIGBiYXRjaF9zaXplICogbl9ldmlkZW5jZSAqIDEyIGJ5dGVzYCAtLSBhYm91dC"
    "AxMDAgTUIgYXQgdGhlIGRlZmF1bHQuCgogICAgICAgIEJhdGNoaW5nIGRvZXMgbm90IGNoYW5nZSB0aGUgcmVzdWx0OiBl"
    "YWNoIHF1ZXJ5J3MgcmFua2luZyBkZXBlbmRzIG9ubHkKICAgICAgICBvbiBpdHMgb3duIHJvdy4gRnVsbCBzdGFibGUgYX"
    "Jnc29ydCBpcyByZXRhaW5lZCB3aXRoaW4gZWFjaCBiYXRjaAogICAgICAgIHJhdGhlciB0aGFuIGFyZ3BhcnRpdGlvbiwg"
    "c28gdGllLWJyZWFraW5nIHJlbWFpbnMgZXhhY3RseSAiaGlnaGVzdAogICAgICAgIHNjb3JlIGZpcnN0LCB0aWVzIGJ5IG"
    "FzY2VuZGluZyBpbmRleCIgYW5kIHRoZSBicnV0ZS1mb3JjZSBlcXVpdmFsZW5jZQogICAgICAgIHRlc3Qgc3RpbGwgaG9s"
    "ZHMuCiAgICAgICAgIiIiCiAgICAgICAgcSA9IGwyX25vcm1hbGl6ZShxdWVyaWVzKSBpZiBzZWxmLm5vcm1hbGl6ZWQgZW"
    "xzZSBucC5hc2FycmF5KHF1ZXJpZXMsIGR0eXBlPW5wLmZsb2F0MzIpCiAgICAgICAgcSA9IG5wLmFzY29udGlndW91c2Fy"
    "cmF5KHEsIGR0eXBlPW5wLmZsb2F0MzIpCiAgICAgICAgbl9xID0gcS5zaGFwZVswXQogICAgICAgIGsgPSBtaW4oaW50KH"
    "RvcF9rKSwgbGVuKHNlbGYuZXZpZGVuY2VfaWRzKSkKICAgICAgICBpZiBrID09IDAgb3Igbl9xID09IDA6CiAgICAgICAg"
    "ICAgIHJldHVybiBbW10gZm9yIF8gaW4gcmFuZ2Uobl9xKV0sIG5wLnplcm9zKChuX3EsIDApLCBkdHlwZT1ucC5mbG9hdD"
    "MyKQoKICAgICAgICBpZiBzZWxmLnVzZV9mYWlzcyBhbmQgc2VsZi5faW5kZXggaXMgbm90IE5vbmU6CiAgICAgICAgICAg"
    "IHNjb3JlcywgaWR4ID0gc2VsZi5faW5kZXguc2VhcmNoKHEsIGspCiAgICAgICAgZWxzZToKICAgICAgICAgICAgYnMgPS"
    "BtYXgoMSwgaW50KGJhdGNoX3NpemUpKQogICAgICAgICAgICBpZHggPSBucC5lbXB0eSgobl9xLCBrKSwgZHR5cGU9bnAu"
    "aW50NjQpCiAgICAgICAgICAgIHNjb3JlcyA9IG5wLmVtcHR5KChuX3EsIGspLCBkdHlwZT1ucC5mbG9hdDMyKQogICAgIC"
    "AgICAgICBmb3Igc3RhcnQgaW4gcmFuZ2UoMCwgbl9xLCBicyk6CiAgICAgICAgICAgICAgICBzdG9wID0gbWluKHN0YXJ0"
    "ICsgYnMsIG5fcSkKICAgICAgICAgICAgICAgIHNpbXMgPSBxW3N0YXJ0OnN0b3BdIEAgc2VsZi52ZWN0b3JzLlQKICAgIC"
    "AgICAgICAgICAgICMgc3RhYmxlIGFyZ3NvcnQgb24gLXNpbXMga2VlcHMgZXF1YWwgc2NvcmVzIGluIGluZGV4IG9yZGVy"
    "CiAgICAgICAgICAgICAgICBvcmRlciA9IG5wLmFyZ3NvcnQoLXNpbXMsIGF4aXM9MSwga2luZD0ic3RhYmxlIilbOiwgOm"
    "tdCiAgICAgICAgICAgICAgICBpZHhbc3RhcnQ6c3RvcF0gPSBvcmRlcgogICAgICAgICAgICAgICAgc2NvcmVzW3N0YXJ0"
    "OnN0b3BdID0gbnAudGFrZV9hbG9uZ19heGlzKHNpbXMsIG9yZGVyLCBheGlzPTEpCiAgICAgICAgICAgICAgICBkZWwgc2"
    "ltcywgb3JkZXIKCiAgICAgICAgaWRzID0gW1tzZWxmLmV2aWRlbmNlX2lkc1tqXSBmb3IgaiBpbiByb3cgaWYgaiA+PSAw"
    "XSBmb3Igcm93IGluIGlkeF0KICAgICAgICByZXR1cm4gaWRzLCBucC5hc2FycmF5KHNjb3JlcywgZHR5cGU9bnAuZmxvYX"
    "QzMikKCiAgICBkZWYgc2F2ZShzZWxmLCBvdXRfZGlyOiBQYXRoKSAtPiBkaWN0W3N0ciwgc3RyXToKICAgICAgICBvdXRf"
    "ZGlyID0gUGF0aChvdXRfZGlyKQogICAgICAgIG91dF9kaXIubWtkaXIocGFyZW50cz1UcnVlLCBleGlzdF9vaz1UcnVlKQ"
    "ogICAgICAgIHZlY190bXAgPSBvdXRfZGlyIC8gInZlY3RvcnMudG1wLm5weSIKICAgICAgICBucC5zYXZlKHZlY190bXAs"
    "IHNlbGYudmVjdG9ycykKICAgICAgICB2ZWNfdG1wLnJlcGxhY2Uob3V0X2RpciAvICJ2ZWN0b3JzLm5weSIpCiAgICAgIC"
    "AgbWV0YV90bXAgPSBvdXRfZGlyIC8gImluZGV4X21ldGEudG1wLmpzb24iCiAgICAgICAgbWV0YV90bXAud3JpdGVfdGV4"
    "dChqc29uLmR1bXBzKHsKICAgICAgICAgICAgImV2aWRlbmNlX2lkcyI6IHNlbGYuZXZpZGVuY2VfaWRzLCAiZGltIjogc2"
    "VsZi5kaW0sCiAgICAgICAgICAgICJub3JtYWxpemVkIjogc2VsZi5ub3JtYWxpemVkLCAibiI6IGxlbihzZWxmLmV2aWRl"
    "bmNlX2lkcyksCiAgICAgICAgICAgICJiYWNrZW5kIjogImZhaXNzIiBpZiBzZWxmLnVzZV9mYWlzcyBlbHNlICJudW1weS"
    "IsCiAgICAgICAgfSwgZW5zdXJlX2FzY2lpPUZhbHNlKSwgZW5jb2Rpbmc9InV0Zi04IikKICAgICAgICBtZXRhX3RtcC5y"
    "ZXBsYWNlKG91dF9kaXIgLyAiaW5kZXhfbWV0YS5qc29uIikKICAgICAgICByZXR1cm4geyJ2ZWN0b3JzIjogc3RyKG91dF"
    "9kaXIgLyAidmVjdG9ycy5ucHkiKSwKICAgICAgICAgICAgICAgICJtZXRhIjogc3RyKG91dF9kaXIgLyAiaW5kZXhfbWV0"
    "YS5qc29uIil9CgogICAgQGNsYXNzbWV0aG9kCiAgICBkZWYgbG9hZChjbHMsIG91dF9kaXI6IFBhdGgsIHVzZV9mYWlzcz"
    "ogYm9vbCB8IE5vbmUgPSBOb25lKSAtPiAiRXZpZGVuY2VJbmRleCI6CiAgICAgICAgb3V0X2RpciA9IFBhdGgob3V0X2Rp"
    "cikKICAgICAgICBtZXRhID0ganNvbi5sb2Fkcygob3V0X2RpciAvICJpbmRleF9tZXRhLmpzb24iKS5yZWFkX3RleHQoZW"
    "5jb2Rpbmc9InV0Zi04IikpCiAgICAgICAgdmVjdG9ycyA9IG5wLmxvYWQob3V0X2RpciAvICJ2ZWN0b3JzLm5weSIpCiAg"
    "ICAgICAgcmV0dXJuIGNscyhtZXRhWyJldmlkZW5jZV9pZHMiXSwgdmVjdG9ycywKICAgICAgICAgICAgICAgICAgIG5vcm"
    "1hbGl6ZT1ib29sKG1ldGEuZ2V0KCJub3JtYWxpemVkIiwgVHJ1ZSkpLAogICAgICAgICAgICAgICAgICAgdmVjdG9yc19w"
    "cmVub3JtYWxpemVkPVRydWUsICAgIyBzdG9yZWQgdmVjdG9ycyBhbHJlYWR5IHVuaXQtbGVuZ3RoCiAgICAgICAgICAgIC"
    "AgICAgICB1c2VfZmFpc3M9dXNlX2ZhaXNzKQoKCmRlZiBicnV0ZV9mb3JjZV9zZWFyY2gocXVlcmllczogbnAubmRhcnJh"
    "eSwgdmVjdG9yczogbnAubmRhcnJheSwKICAgICAgICAgICAgICAgICAgICAgICBldmlkZW5jZV9pZHM6IFNlcXVlbmNlW3"
    "N0cl0sIHRvcF9rOiBpbnQKICAgICAgICAgICAgICAgICAgICAgICApIC0+IHR1cGxlW2xpc3RbbGlzdFtzdHJdXSwgbnAu"
    "bmRhcnJheV06CiAgICAiIiJSZWZlcmVuY2UgaW1wbGVtZW50YXRpb24gdXNlZCBvbmx5IGJ5IHRlc3RzIHRvIHByb3ZlIH"
    "RoZSBpbmRleCBleGFjdC4iIiIKICAgIHEsIHYgPSBsMl9ub3JtYWxpemUocXVlcmllcyksIGwyX25vcm1hbGl6ZSh2ZWN0"
    "b3JzKQogICAgc2ltcyA9IHEgQCB2LlQKICAgIGsgPSBtaW4odG9wX2ssIGxlbihldmlkZW5jZV9pZHMpKQogICAgaWR4ID"
    "0gbnAuYXJnc29ydCgtc2ltcywgYXhpcz0xLCBraW5kPSJzdGFibGUiKVs6LCA6a10KICAgIHNjb3JlcyA9IG5wLnRha2Vf"
    "YWxvbmdfYXhpcyhzaW1zLCBpZHgsIGF4aXM9MSkKICAgIHJldHVybiBbW2V2aWRlbmNlX2lkc1tqXSBmb3IgaiBpbiByb3"
    "ddIGZvciByb3cgaW4gaWR4XSwgc2NvcmVzCg=="
)

RET.mkdir(parents=True, exist_ok=True)
dest = RET / "faiss_index.py"
old = dest.stat().st_size if dest.exists() else 0

if dest.exists():
    shutil.copy2(dest, RET / "faiss_index.py.backup")
    print("  backed up existing -> faiss_index.py.backup")

tmp = dest.with_suffix(".py.tmp")
tmp.write_bytes(base64.b64decode(B64))
tmp.replace(dest)

got = hashlib.sha256(dest.read_bytes()).hexdigest()
size = dest.stat().st_size
print("  path   :", dest)
print("  size   :", old, "->", size, "bytes (expect", EXPECTED_SIZE, ")")
print("  sha256 :", got)
ok = (got == EXPECTED_SHA and size == EXPECTED_SIZE)
print("  verify :", "OK" if ok else "FAILED")
assert ok, "write verification failed"

for d in (RET, PROJECT / "scripts", PROJECT / "src"):
    pc = d / "__pycache__"
    if pc.exists():
        shutil.rmtree(pc, ignore_errors=True)

txt = dest.read_text(encoding="utf-8")
print()
print("  batching markers:")
for marker in ["batch_size: int = 128", "for start in range(0, n_q, bs)",
               "del sims, order"]:
    print("     ", "FOUND  " if marker in txt else "MISSING", marker)
assert "for start in range(0, n_q, bs)" in txt

print()
print("Fixed. Peak memory for FEVER dense drops from ~12.87 GB to ~103 MB.")
print("Now run FEVER dense retrieval - validation first.")
