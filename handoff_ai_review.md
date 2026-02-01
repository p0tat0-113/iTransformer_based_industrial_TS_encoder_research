# 인수인계 문서 (AI 코드 신뢰성 점검용)

## 0) 목적
이 문서는 **현재 코드가 의도대로 작성되었는지** 및 **실험 결과가 신뢰 가능한지**를 다른 AI가 점검할 수 있도록, 지금까지의 구현 변경과 핵심 컨텍스트를 상세히 전달한다.

---

## 1) 현재 코드 상태 요약
### 핵심 수정/개선 사항
1) **Var‑MAE → downstream 가중치 로드 문제 해결**
- 문제: Var‑MAE 체크포인트 키(`encoder_layers.*`, `encoder_norm.*`, `value_proj.*`)와 downstream 모델 키(`encoder.attn_layers.*`, `encoder.norm.*`, `patch_embed.*`)가 달라, downstream LP에서 대부분 가중치가 **로드되지 않는 버그**가 있었음.
- 해결: `src/itransformer/downstream.py`에 **키 remap 로직** 추가. 또한 patch embedding의 shape mismatch를 막기 위해 **shape 체크 후 필터링**.
- 결과: LP/FT에서 encoder가 실제로 SSL pretrain 가중치를 사용하도록 개선.

2) **P1 mean_pool patch_len 강제 변경 버그 제거**
- 문제: downstream 쪽에서 mean_pool인 경우 `patch_len = seq_len`로 강제하여, P1에서 patch_len sweep 의미 상실.
- 해결: `_maybe_inject_patch_len`에서 해당 강제 로직 제거.

3) **Patch 모델 tokenization 개선**
- `src/itransformer/models/patch_transformer.py`
  - 기존: patch 평균(`patch_mean`) 후 `value_proj(1→d)`
  - 변경: patch 벡터(`patch_len`) → `patch_embed(patch_len→d)`
  - mean_pool은 patch embedding 후 **patch축 평균**으로 처리.

4) **Var‑MAE patch 경로 수정**
- `src/itransformer/ssl/var_mae.py`
  - patch path에서 **patch_embed(patch_len→d)** 사용
  - patch_mode가 mean_pool인 경우: patch embedding 후 patch축 평균
  - patch_mode가 mean_pool이 아닌 경우: patch tokens 유지하고 encoder 통과 후 patch축 평균
  - **patch_mixer(MLP)** 옵션 추가 (patch 축 정보 혼합)

---

## 2) 현재 핵심 파일 변경 사항
- `src/itransformer/downstream.py`
  - `_remap_ssl_state()` 추가: Var‑MAE ckpt 키를 downstream 모델 키로 변환
  - shape mismatch 시 스킵
  - mean_pool에서 patch_len 강제 제거
  - patch 모델 LP freeze 대상 수정 (`patch_embed` 포함)

- `src/itransformer/models/patch_transformer.py`
  - patch embedding 구조 변경 (patch_len 차이를 실제 반영)

- `src/itransformer/ssl/var_mae.py`
  - patch embedding + patch_mixer 적용
  - patch tokenization 및 평균 처리 로직 수정

---

## 3) 실험군 B(B_urgent) 계획
`conf/plan/B_urgent.yaml`
- pretrain: Mix(ETTh1 + Exchange), sample_mode=balanced
- baseline: **P0는 Var‑MAE**
- patch 모델: **P1~P4는 Patch‑MAE**
- patch_len=16, seed=0, epochs=10
- downstream: ETTh1, LP 설정
- agg: B‑EV‑2 / B‑EV‑4 포함

---

## 4) 주요 관찰된 문제 (신뢰성 관련 이슈)
### (A) P2/P3/P4 결과가 거의 동일하게 보이는 문제
- 겉보기에는 소수점 한자리 수준에서 거의 동일
- 원인 후보:
  1. **Patch attention 차이가 실제로 작은 설정** (local_win=2, patch_count가 작으면 full attention에 가까움)
  2. Var‑MAE 구조에서 patch 평균 및 patch_mixer가 차이를 약화시키는 효과
  3. 짧은 epoch, 동일 seed, 동일 데이터로 인한 결과 수렴

### (B) Downstream 가중치 로드 실패 문제
- 과거 버전은 거의 전부 mismatch로 로드 실패 → 결과 왜곡 가능
- 현재 버전은 remap 적용됨.

---

## 5) Agg 결과 및 Figure 생성 위치
### Agg
- `artifacts/agg/AGG.ETTh1.B-EV-2__ON=/agg.json`
- `artifacts/agg/AGG.Mix.B-EV-4__ON=/agg.json`

### Figures
- Pretrain train loss:
  - `artifacts/figures/B_urgent_pretrain/train_loss_all.png`
  - 개별 + grad_norm:
    - `artifacts/figures/B_urgent_pretrain/train_loss_grad_B-URG-TR.Mix.P*.png`

- Downstream train/val loss + test mse/mae:
  - `artifacts/figures/B_urgent_downstream/downstream_B-URG-DS.ETTh1.P*.png`

- Agg figure:
  - `artifacts/figures/B_urgent_agg/B-EV-2_test_metrics.png`
  - `artifacts/figures/B_urgent_agg/B-EV-2_cost_vs_mse.png`
  - `artifacts/figures/B_urgent_agg/B-EV-4_cka.png`

---

## 6) 실행/재현 명령
### B_urgent 전체 실행
```
PYTHONPATH=/workspace/src python -m itransformer.orchestrator.run plan=conf/plan/B_urgent.yaml
```

### 개별 점검
- pretrain: `python -m itransformer.pretrain ...`
- downstream: `python -m itransformer.downstream ...`
- agg: `python -m itransformer.analysis_entry data=... analysis.code=...`

---

## 7) 리뷰 시 확인해야 할 핵심 포인트
1. **Var‑MAE ckpt → downstream 로드**
   - 로드 로그에서 unexpected/missing이 projector 정도만 남는지 확인
2. **P1 mean_pool에서 patch_len이 실제로 반영되는지**
   - `patch_embed` 구조에서 patch_len 차이가 의미 있게 나타나는지 확인
3. **Patch-MAE + patch_len**
   - patch_len 변경이 실제 loss/curve에 영향을 주는지 점검
4. **P2/P3/P4 차이가 충분히 드러나는지**
   - local_win/patch_count 설정 재검토 필요성

---

## 8) 현재 커밋
- `a7c6900 Fix SSL checkpoint remap and update B_urgent plan`

---

## 9) 추가 메모
- `artifacts/` 디렉터리는 untracked 상태 (실험 결과 보존 목적)
- 실험군 A/C 관련 구현은 별도 문서 `documents/`에 정리됨

