import re
import logging
from typing import List, Dict, Optional

class KtpDataExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Mapping of common province names to their Indonesian NIK codes (first 2 digits)
        self.province_codes = {
            "ACEH": "11", "SUMATERAUTARA": "12", "SUMATERABARAT": "13", "RIAU": "14",
            "JAMBI": "15", "SUMATERASELATAN": "16", "BENGKULU": "17", "LAMPUNG": "18",
            "BANGKABELITUNG": "19", "KEPULAUANRIAU": "21", "DKIJAKARTA": "31",
            "JAWABARAT": "32", "JAWATENGAH": "33", "DIYOGYAKARTA": "34", "JAWATIMUR": "35",
            "BANTEN": "36", "BALI": "51", "NUSATENGGARABARAT": "52", "NUSATENGGARATIMUR": "53",
            "KALIMANTANBARAT": "61", "KALIMANTANTENGAH": "62", "KALIMANTANSELATAN": "63",
            "KALIMANTANTIMUR": "64", "KALIMANTANUTARA": "65", "SULAWESIUTARA": "71",
            "SULAWESITENGAH": "72", "SULAWESISELATAN": "73", "SULAWESITENGGARA": "74",
            "GORONTALO": "75", "SULAWESIBARAT": "76", "MALUKU": "81", "MALUKUUTARA": "82",
            "PAPUABARAT": "91", "PAPUA": "92"
        }

    def extract_nik(self, text_blocks: List[Dict]) -> Dict:
        """
        Robustly extracts NIK by cross-validating candidates against secondary fields
        (Birthdate, Gender, Province) and applying fuzzy corrections.
        """
        # 1. Extract secondary fields for validation
        fields = self._extract_secondary_fields(text_blocks)
        self.logger.info(f"Secondary fields used for validation: {fields}")

        # 2. Gather all candidates (16-digit-like patterns)
        candidates = self._get_nik_candidates(text_blocks)
        
        if not candidates:
            return {"nik": None, "confidence": 0.0, "method": "none"}

        # 3. Score and repair candidates
        scored_candidates = []
        for cand in candidates:
            repaired_nik, score, method = self._evaluate_and_repair(cand, fields)
            scored_candidates.append({
                "nik": repaired_nik,
                "confidence": score / 100.0,
                "method": method,
                "score": score,
                "box": cand.get('box')
            })

        # 4. Pick the best result
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        best = scored_candidates[0]

        # Final quality gate
        if best['score'] < 30:
            self.logger.warning("Low confidence NIK extraction. Result may be inaccurate.")

        return {
            "nik": best['nik'],
            "confidence": min(best['confidence'], 1.0),
            "method": best['method'],
            "box": best.get('box')
        }

    def _extract_secondary_fields(self, text_blocks: List[Dict]) -> Dict:
        """Finds Birthdate, Gender, and Province in the OCR list."""
        all_text = " ".join([b['text'].upper() for b in text_blocks])
        all_text_no_space = all_text.replace(" ", "").replace(":", "").replace(".", "")

        fields = {
            "birthdate": None,  # (DD, MM, YY)
            "gender": "UNKNOWN",
            "province_code": None
        }

        # 1. Extract Birthdate (e.g., 05-12-1987)
        # We look for DD-MM-YYYY or DD.MM.YYYY
        date_match = re.search(r'(\d{2})[-.](\d{2})[-.](\d{4})', all_text)
        if date_match:
            fields["birthdate"] = (date_match.group(1), date_match.group(2), date_match.group(3)[2:])

        # 2. Extract Gender
        if "PEREMPUAN" in all_text:
            fields["gender"] = "FEMALE"
        elif "LAKI" in all_text or "LAKILAKI" in all_text:
            fields["gender"] = "MALE"

        # 3. Extract Province Code
        for prov_name, code in self.province_codes.items():
            if prov_name in all_text_no_space:
                fields["province_code"] = code
                break

        return fields

    def _get_nik_candidates(self, text_blocks: List[Dict]) -> List[Dict]:
        """Finds all potential 16-digit strings."""
        candidates = []
        for block in text_blocks:
            text = block['text'].replace(" ", "").replace(":", "")
            # Find any sequence of 15-17 digits (to allow for minor OCR jitter)
            matches = re.findall(r'\d{15,17}', text)
            for m in matches:
                candidates.append({
                    "raw": m[:16].zfill(16),
                    "base_confidence": block.get('confidence', 0.5),
                    "box": block.get('box')
                })
        
        # Fallback: Concatenate all numeric parts
        if not candidates:
            full_nums = "".join(re.findall(r'\d+', " ".join([b['text'] for b in text_blocks])))
            if len(full_nums) >= 16:
                candidates.append({
                    "raw": full_nums[:16],
                    "base_confidence": 0.4,
                    "box": None
                })

        return candidates

    def _evaluate_and_repair(self, candidate: Dict, fields: Dict) -> (str, float, str):
        """Scores a NIK candidate and attempts logic-based repairs."""
        raw_nik = candidate['raw']
        score = candidate['base_confidence'] * 20 # Start with OCR confidence base (0-20 pts)
        method = "ocr_direct"
        
        # 1. Check Birthdate + Gender (The strongest validator)
        # Digits 7-12: DD MM YY
        if fields['birthdate']:
            expected_dd = int(fields['birthdate'][0])
            if fields['gender'] == "FEMALE":
                expected_dd += 40
            
            nik_dd = int(raw_nik[6:8])
            nik_mm = raw_nik[8:10]
            nik_yy = raw_nik[10:12]

            if nik_dd == expected_dd and nik_mm == fields['birthdate'][1] and nik_yy == fields['birthdate'][2]:
                score += 60 # Major boost for birthdate match
                method += "_birthdate_validated"
            elif nik_mm == fields['birthdate'][1] and nik_yy == fields['birthdate'][2]:
                # Partially matches month/year, maybe day is misread?
                score += 30
                method += "_birthdate_partial"

        # 2. Check Province Code
        if fields['province_code']:
            if raw_nik[:2] == fields['province_code']:
                score += 20
                method += "_province_validated"
            else:
                # 3. FUZZY REPAIR: If it matches birthdate but province is a common flip
                # e.g. Riau is 14. OCR read 74 (where 7 is misread 1).
                repaired_nik = self._attempt_fuzzy_repair(raw_nik, fields['province_code'])
                if repaired_nik != raw_nik:
                    # After repair, it matches!
                    score += 15 # Slightly lower than exact match
                    raw_nik = repaired_nik
                    method += "_repaired_from_" + candidate['raw'][:1]

        return raw_nik, score, method

    def _attempt_fuzzy_repair(self, nik: str, expected_prov: str) -> str:
        """Fixes common OCR digit flips specifically for the province code."""
        # Mapping of (OCR Read) -> (Likely Correct)
        flips = {'7': '1', '1': '7', '8': 'B', '3': '8', '5': 'S', '0': 'D'}
        
        current_prov = nik[:2]
        if current_prov == expected_prov:
            return nik
            
        # Try flipping the first digit if it matches the expected province code
        first_digit = nik[0]
        if first_digit in flips and (flips[first_digit] + nik[1]) == expected_prov:
            return flips[first_digit] + nik[1:]
            
        return nik

    def _is_on_same_line(self, box1, box2, tolerance=30):
        # Center Y of both boxes
        y1 = (box1[0][1] + box1[2][1]) / 2
        y2 = (box1[0][1] + box1[2][1]) / 2
        return abs(y1 - y2) < tolerance
