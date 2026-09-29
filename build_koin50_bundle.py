import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path('downloaded_shards')
OUT=Path('koin50_bundle')
OUT.mkdir(exist_ok=True)
shards=[]
for s in range(10):
    found=list(ROOT.rglob(f'summary_shard_{s:02d}.json'))
    models=list(ROOT.rglob(f'koin50_shard_{s:02d}.pt'))
    samples=list(ROOT.rglob(f'samples_shard_{s:02d}.png'))
    logs=list(ROOT.rglob(f'training_log_{s:02d}.json'))
    if not (found and models and samples and logs):
        raise RuntimeError(f'missing shard {s}')
    summary=json.loads(found[0].read_text())
    shutil.copy2(models[0],OUT/models[0].name)
    shutil.copy2(samples[0],OUT/samples[0].name)
    shutil.copy2(logs[0],OUT/logs[0].name)
    shutil.copy2(found[0],OUT/found[0].name)
    shards.append(summary)

covered=[]
for x in shards: covered.extend(x['global_class_ids'])
if covered != [f'{i:04d}' for i in range(50)]:
    raise RuntimeError(f'class coverage mismatch: {covered}')

router={
    'system':'KoIn50 Korean celebrity sharded conditional VAE',
    'global_identity_ids':covered,
    'routing_rule':'global class 0000-0049 -> shard=floor(class_id/5), local_id=class_id mod 5',
    'shards':[
        {'shard':x['shard'],'classes':x['global_class_ids'],'model':f"koin50_shard_{x['shard']:02d}.pt"}
        for x in shards
    ],
    'raw_source_photos_in_bundle':False,
    'usage_scope':'KoIn official README states academic purposes',
}
(OUT/'router.json').write_text(json.dumps(router,indent=2),encoding='utf-8')

aggregate={
    'identity_classes':50,
    'total_accepted_face_crops':sum(x['accepted_images'] for x in shards),
    'total_train_images':sum(x['train_images'] for x in shards),
    'total_validation_images':sum(x['val_images'] for x in shards),
    'mean_val_identity_accuracy':sum(x['val_identity_accuracy'] for x in shards)/10,
    'min_val_identity_accuracy':min(x['val_identity_accuracy'] for x in shards),
    'max_val_identity_accuracy':max(x['val_identity_accuracy'] for x in shards),
    'mean_val_reconstruction_l1':sum(x['val_reconstruction_l1'] for x in shards)/10,
    'total_model_parameters_across_shards':sum(x['parameters'] for x in shards),
    'per_shard':shards,
    'raw_source_photos_uploaded':False,
    'usage_scope':'KoIn official README: academic purposes',
}
(OUT/'aggregate_summary.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')

# 10 rows x 5 identities: combine per-shard generation strips.
imgs=[]
for s in range(10):
    p=OUT/f'samples_shard_{s:02d}.png'
    imgs.append(Image.open(p).convert('RGB'))
w=max(i.width for i in imgs); h=sum(i.height for i in imgs)
canvas=Image.new('RGB',(w,h),'white'); y=0
for im in imgs:
    canvas.paste(im,(0,y));y+=im.height
canvas.save(OUT/'generated_50_identity_montage.png')
print(json.dumps(aggregate,indent=2))
