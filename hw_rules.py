"""hw_rules.py — auto-generated from train_dt.py, do not edit manually"""

def predict_action(tok0, tok1, tok2, n_active, both_idle, c2_rank, c3_rank):
    """Return action label: 0=PAIR 1=SPLIT 2=SINGLE_C2 3=SINGLE_C3 4=PREFETCH"""
    tok_diff = tok0 - tok1
    tok_ratio4 = 1 if tok0 >= 4 else 0
    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,
               tok_ratio4=tok_ratio4, n_active=n_active,
               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)
    if row['tok_diff'] <= 5.5:
        if row['tok1'] <= 0.5:
            if row['tok0'] <= 3.5:
                if row['both_idle'] <= 0.5:
                    if row['tok0'] <= 2.5:
                        return 2  # SINGLE_C2 n=759
                    else:
                        return 2  # SINGLE_C2 n=46
                else:
                    if row['tok0'] <= 2.5:
                        return 2  # SINGLE_C2 n=1169
                    else:
                        return 1  # SPLIT n=20
            else:
                if row['both_idle'] <= 0.5:
                    if row['tok0'] <= 4.5:
                        return 2  # SINGLE_C2 n=11
                    else:
                        return 1  # SPLIT n=92
                else:
                    return 1  # SPLIT n=615
        else:
            if row['tok2'] <= 4.5:
                if row['both_idle'] <= 0.5:
                    if row['tok2'] <= 0.5:
                        return 0  # PAIR n=1432
                    else:
                        return 3  # SINGLE_C3 n=674
                else:
                    if row['tok0'] <= 4.5:
                        return 0  # PAIR n=4682
                    else:
                        return 0  # PAIR n=3570
            else:
                if row['both_idle'] <= 0.5:
                    if row['tok1'] <= 8.5:
                        return 3  # SINGLE_C3 n=486
                    else:
                        return 1  # SPLIT n=487
                else:
                    if row['c3_rank'] <= 0.5:
                        return 2  # SINGLE_C2 n=667
                    else:
                        return 0  # PAIR n=1151
    else:
        if row['tok2'] <= 0.5:
            if row['both_idle'] <= 0.5:
                if row['tok_diff'] <= 6.5:
                    if row['tok0'] <= 6.5:
                        return 1  # SPLIT n=210
                    else:
                        return 2  # SINGLE_C2 n=189
                else:
                    if row['tok1'] <= 2.5:
                        return 1  # SPLIT n=1772
                    else:
                        return 1  # SPLIT n=709
            else:
                if row['tok_diff'] <= 6.5:
                    if row['tok0'] <= 8.5:
                        return 1  # SPLIT n=190
                    else:
                        return 1  # SPLIT n=119
                else:
                    if row['tok1'] <= 0.5:
                        return 1  # SPLIT n=2327
                    else:
                        return 1  # SPLIT n=492
        else:
            if row['both_idle'] <= 0.5:
                if row['c2_rank'] <= 3.5:
                    if row['tok1'] <= 9.5:
                        return 1  # SPLIT n=66
                    else:
                        return 1  # SPLIT n=51
                else:
                    if row['n_active'] <= 4.5:
                        return 3  # SINGLE_C3 n=2815
                    else:
                        return 3  # SINGLE_C3 n=1011
            else:
                if row['n_active'] <= 3.5:
                    if row['tok0'] <= 8.5:
                        return 2  # SINGLE_C2 n=151
                    else:
                        return 1  # SPLIT n=718
                else:
                    if row['n_active'] <= 5.5:
                        return 2  # SINGLE_C2 n=1593
                    else:
                        return 0  # PAIR n=1582

def predict_c2_s1(tok0, tok1, tok2, n_active, both_idle, c2_rank, c3_rank):
    """Return shape label: 0=ShapeA 1=ShapeB 2=ShapeC"""
    tok_diff = tok0 - tok1
    tok_ratio4 = 1 if tok0 >= 4 else 0
    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,
               tok_ratio4=tok_ratio4, n_active=n_active,
               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)
    if row['tok0'] <= 12.5:
        if row['tok1'] <= 6.5:
            if row['tok_diff'] <= 0.5:
                if row['c2_rank'] <= 3.5:
                    return 2  # ShapeC n=444
                else:
                    return 1  # ShapeB n=4064
            else:
                if row['tok0'] <= 1.5:
                    return 2  # ShapeC n=1240
                else:
                    return 1  # ShapeB n=9677
        else:
            if row['tok2'] <= 1:
                if row['tok_diff'] <= 2.5:
                    return 0  # ShapeA n=859
                else:
                    return 0  # ShapeA n=62
            else:
                if row['c3_rank'] <= 4:
                    return 1  # ShapeB n=664
                else:
                    return 1  # ShapeB n=947
    else:
        if row['n_active'] <= 3.5:
            if row['tok2'] <= 0.5:
                if row['tok1'] <= 2.5:
                    return 0  # ShapeA n=2107
                else:
                    return 0  # ShapeA n=1649
            else:
                if row['tok1'] <= 2.5:
                    return 2  # ShapeC n=184
                else:
                    return 0  # ShapeA n=1396
        else:
            if row['tok2'] <= 4.5:
                if row['n_active'] <= 6.5:
                    return 2  # ShapeC n=1545
                else:
                    return 1  # ShapeB n=492
            else:
                if row['tok2'] <= 6.5:
                    return 1  # ShapeB n=558
                else:
                    return 0  # ShapeA n=700

def predict_c2_s3(tok0, tok1, tok2, n_active, both_idle, c2_rank, c3_rank):
    """Return shape label: 0=ShapeA 1=ShapeB 2=ShapeC"""
    tok_diff = tok0 - tok1
    tok_ratio4 = 1 if tok0 >= 4 else 0
    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,
               tok_ratio4=tok_ratio4, n_active=n_active,
               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)
    if row['tok0'] <= 8.5:
        if row['tok1'] <= 0.5:
            if row['tok0'] <= 2.5:
                return 2  # ShapeC n=1565
            else:
                return 1  # ShapeB n=1758
        else:
            if row['tok1'] <= 6.5:
                if row['tok_diff'] <= 3.5:
                    return 1  # ShapeB n=6582
                else:
                    return 1  # ShapeB n=2727
            else:
                if row['tok2'] <= 1:
                    return 0  # ShapeA n=357
                else:
                    return 1  # ShapeB n=518
    else:
        if row['n_active'] <= 3.5:
            if row['tok0'] <= 12.5:
                if row['tok_diff'] <= 4.5:
                    return 0  # ShapeA n=825
                else:
                    return 1  # ShapeB n=1612
            else:
                if row['tok2'] <= 0.5:
                    return 0  # ShapeA n=3567
                else:
                    return 0  # ShapeA n=1475
        else:
            if row['tok2'] <= 2.5:
                if row['n_active'] <= 6.5:
                    return 2  # ShapeC n=1464
                else:
                    return 1  # ShapeB n=400
            else:
                if row['tok2'] <= 8.5:
                    return 1  # ShapeB n=2066
                else:
                    return 0  # ShapeA n=779

def predict_c3_s1(tok0, tok1, tok2, n_active, both_idle, c2_rank, c3_rank):
    """Return shape label: 0=ShapeA 1=ShapeB 2=ShapeC"""
    tok_diff = tok0 - tok1
    tok_ratio4 = 1 if tok0 >= 4 else 0
    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,
               tok_ratio4=tok_ratio4, n_active=n_active,
               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)
    if row['tok0'] <= 12.5:
        if row['tok1'] <= 4.5:
            if row['c2_rank'] <= 3.5:
                if row['tok0'] <= 4.5:
                    return 2  # ShapeC n=622
                else:
                    return 1  # ShapeB n=671
            else:
                if row['both_idle'] <= 0.5:
                    return 1  # ShapeB n=2912
                else:
                    return 1  # ShapeB n=7712
        else:
            if row['tok2'] <= 0.5:
                if row['tok1'] <= 6.5:
                    return 1  # ShapeB n=735
                else:
                    return 0  # ShapeA n=1056
            else:
                if row['c3_rank'] <= 4:
                    return 1  # ShapeB n=680
                else:
                    return 1  # ShapeB n=2283
    else:
        if row['n_active'] <= 3.5:
            if row['both_idle'] <= 0.5:
                if row['tok2'] <= 0.5:
                    return 0  # ShapeA n=1454
                else:
                    return 0  # ShapeA n=1023
            else:
                if row['tok_diff'] <= 1.5:
                    return 0  # ShapeA n=556
                else:
                    return 0  # ShapeA n=2523
        else:
            if row['both_idle'] <= 0.5:
                if row['tok2'] <= 2.5:
                    return 2  # ShapeC n=695
                else:
                    return 1  # ShapeB n=746
            else:
                if row['n_active'] <= 5.5:
                    return 0  # ShapeA n=620
                else:
                    return 1  # ShapeB n=617

def predict_c3_s3(tok0, tok1, tok2, n_active, both_idle, c2_rank, c3_rank):
    """Return shape label: 0=ShapeA 1=ShapeB 2=ShapeC"""
    tok_diff = tok0 - tok1
    tok_ratio4 = 1 if tok0 >= 4 else 0
    row = dict(tok0=tok0, tok1=tok1, tok2=tok2, tok_diff=tok_diff,
               tok_ratio4=tok_ratio4, n_active=n_active,
               both_idle=both_idle, c2_rank=c2_rank, c3_rank=c3_rank)
    if row['tok0'] <= 12.5:
        if row['both_idle'] <= 0.5:
            if row['c3_rank'] <= 0.5:
                if row['tok2'] <= 2.5:
                    return 0  # ShapeA n=328
                else:
                    return 2  # ShapeC n=348
            else:
                if row['tok0'] <= 2.5:
                    return 2  # ShapeC n=540
                else:
                    return 1  # ShapeB n=3388
        else:
            if row['tok1'] <= 6.5:
                if row['tok_diff'] <= 1.5:
                    return 1  # ShapeB n=4468
                else:
                    return 1  # ShapeB n=5115
            else:
                if row['tok2'] <= 2.5:
                    return 0  # ShapeA n=342
                else:
                    return 1  # ShapeB n=682
    else:
        if row['tok2'] <= 0.5:
            if row['tok1'] <= 2.5:
                if row['tok_diff'] <= 15.5:
                    return 0  # ShapeA n=235
                else:
                    return 0  # ShapeA n=1871
            else:
                if row['tok1'] <= 4.5:
                    return 1  # ShapeB n=125
                else:
                    return 0  # ShapeA n=1349
        else:
            if row['tok2'] <= 2.5:
                if row['both_idle'] <= 0.5:
                    return 2  # ShapeC n=1178
                else:
                    return 1  # ShapeB n=306
            else:
                if row['n_active'] <= 3.5:
                    return 0  # ShapeA n=1020
                else:
                    return 1  # ShapeB n=1533
