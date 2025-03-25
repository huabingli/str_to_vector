from fastapi import APIRouter, BackgroundTasks, Request

from core.config import templates
from models.image_vector import ImageSimilarityBatch, ImageSimilarityOutBatch, ImageVector, ImageVectorOut
from schemas.base import R
from utils.images_vector.embedding import async_get_image_embedding
from utils.images_vector.model_factory import LargeModelName
from utils.images_vector.similarity import async_image_calculate_cosine_similarity

router = APIRouter(prefix='/image_vector', tags=['图片向量转换'])


@router.post(
        '/openai/',
        summary='openapi图片转向量',
        response_model=R[ImageVectorOut],
        response_model_exclude_none=True
)
async def get_image_vector(image_url: ImageVector):
    vector = await async_get_image_embedding([image_url.image_url])
    data = ImageVectorOut(vector=vector[0], **image_url.model_dump(exclude_unset=True))
    return R.success(data=data)


@router.post(
        '/google/',
        summary='谷歌模型图片转向量',
        response_model=R[ImageVectorOut],
        response_model_exclude_none=True
)
async def get_image_vector(image_url: ImageVector):
    vector = await async_get_image_embedding([image_url.image_url], LargeModelName.google)
    data = ImageVectorOut(vector=vector[0], **image_url.model_dump(exclude_unset=True))
    return R.success(data=data)


@router.get('/image_similarity/', summary='图片相似度计算')
async def get_image_similarity(request: Request):
    return templates.TemplateResponse(name='image_similarity.html', request=request,
                                      context={'base_url': request.base_url})


def delete_image_tmp(images: ImageSimilarityBatch):
    images.del_image_tmp()
    for image in images.batch:
        image.del_image_tmp()


@router.post('/image_similarity/', summary='openai图片相似度计算', response_model=R[ImageSimilarityOutBatch])
async def get_image_similarity(images: ImageSimilarityBatch, background_tasks: BackgroundTasks):
    data = await async_image_calculate_cosine_similarity(images)
    data.sort_by_similarity()
    background_tasks.add_task(delete_image_tmp, images)
    return R.success(data=data)


@router.post('/google/image_similarity/', summary='谷歌图片相似度计算', response_model=R[ImageSimilarityOutBatch])
async def get_image_similarity(images: ImageSimilarityBatch, background_tasks: BackgroundTasks):
    data = await async_image_calculate_cosine_similarity(images, LargeModelName.google)
    data.sort_by_similarity()
    background_tasks.add_task(delete_image_tmp, images)
    return R.success(data=data)
