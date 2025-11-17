import os
import logging
import argparse
import gc
import random
import time
import tempfile
import asyncio
from threading import Thread

import imageio
import torch
from diffusers.utils import load_image
from PIL import Image

from skyreels_v2_infer.modules import download_model
from skyreels_v2_infer.pipelines import Image2VideoPipeline
from skyreels_v2_infer.pipelines import PromptEnhancer
from skyreels_v2_infer.pipelines import resizecrop
from skyreels_v2_infer.pipelines import Text2VideoPipeline

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

# Bot konfiguratsiyasi
BOT_TOKEN = "8516869371:AAF7i_MsGe95AVaDuQrHzRL_UX408o0qI8w"  # BotFather dan olingan token
ADMIN_IDS = [5792951787, 6499839380]  # Admin ID lari

# Model konfiguratsiyasi
MODEL_ID_CONFIG = {
    "text2video": [
        "Skywork/SkyReels-V2-T2V-14B-540P",
        "Skywork/SkyReels-V2-T2V-14B-720P",
    ],
    "image2video": [
        "Skywork/SkyReels-V2-I2V-1.3B-540P",
        "Skywork/SkyReels-V2-I2V-14B-540P",
        "Skywork/SkyReels-V2-I2V-14B-720P",
    ],
}

# Logging sozlash
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Foydalanuvchi sessiyalari
user_sessions = {}

class UserSession:
    def __init__(self, user_id):
        self.user_id = user_id
        self.model_type = "text2video"
        self.model_id = "Skywork/SkyReels-V2-T2V-14B-540P"
        self.resolution = "540P"
        self.num_frames = 41
        self.guidance_scale = 6.0
        self.shift = 8.0
        self.inference_steps = 30
        self.fps = 24
        self.seed = random.randint(0, 4294967294)
        self.prompt = ""
        self.image_path = None
        self.use_usp = False
        self.offload = False
        self.prompt_enhancer = False
        self.teacache = False
        self.teacache_thresh = 0.2
        self.use_ret_steps = False
        self.is_generating = False

def get_user_session(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(user_id)
    return user_sessions[user_id]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Salom {user.first_name}!\n\n"
        "🎬 SkyReels Video Generator botiga xush kelibsiz!\n\n"
        "Bu bot yordamida matn yoki rasm asosida video yaratishingiz mumkin.\n\n"
        "📖 **Qo'llanma:**\n"
        "1. /generate - Video yaratishni boshlash\n"
        "2. /settings - Sozlamalarni o'zgartirish\n"
        "3. /status - Joriy holatni ko'rish\n"
        "4. /help - Yordam olish\n\n"
        "⚠️ **Eslatma:** Video yaratish bir necha daqiqa vaqt olishi mumkin.",
        reply_markup=main_menu_keyboard()
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """
🤖 **SkyReels Bot Yordami**

🎥 **Video Yaratish:**
- Matn asosida video yaratish uchun /generate buyrug'ini bering
- Rasm asosida video yaratish uchun rasm yuboring va prompt kiriting

⚙️ **Sozlamalar:**
- /settings - Barcha sozlamalarni o'zgartirish
- /status - Joriy sozlamalarni ko'rish

📊 **Parametrlar:**
- **Resolution:** 540P yoki 720P
- **Frames:** 10-200 (odatda 41)
- **Guidance Scale:** 1.0-20.0 (6.0 optimal)
- **Inference Steps:** 1-100 (30 optimal)

⏳ **Vaqt:** Video yaratish 2-10 daqiqa davom etadi

📝 **Prompt Maslahatlari:**
- Batafsil va aniq tasvirlaring
- Harakatni ko'rsating
- Sifatli so'zlardan foydalaning
    """
    await update.message.reply_text(help_text)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    status_text = f"""
📊 **Joriy Sozlamalar:**

🎯 **Model Turi:** {session.model_type}
🆔 **Model:** {session.model_id.split('/')[-1]}
📺 **Resolution:** {session.resolution}
🎞️ **Kadrlar Soni:** {session.num_frames}
🎛️ **Guidance Scale:** {session.guidance_scale}
⚡ **Shift:** {session.shift}
🔢 **Inference Steps:** {session.inference_steps}
🎬 **FPS:** {session.fps}
🌱 **Seed:** {session.seed}
📝 **Prompt Enhancer:** {'✅' if session.prompt_enhancer else '❌'}
⚡ **TeaCache:** {'✅' if session.teacache else '❌'}

⚙️ Sozlamalarni o'zgartirish uchun /settings buyrug'idan foydalaning.
    """
    await update.message.reply_text(status_text)

def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎬 Video Yaratish", callback_data="generate")],
        [InlineKeyboardButton("⚙️ Sozlamalar", callback_data="settings")],
        [InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("❓ Yordam", callback_data="help")]
    ]
    return InlineKeyboardMarkup(keyboard)

def settings_keyboard():
    keyboard = [
        [InlineKeyboardButton("🎯 Model Turi", callback_data="setting_model_type")],
        [InlineKeyboardButton("🆔 Model", callback_data="setting_model")],
        [InlineKeyboardButton("📺 Resolution", callback_data="setting_resolution")],
        [InlineKeyboardButton("🎞️ Kadrlar Soni", callback_data="setting_frames")],
        [InlineKeyboardButton("🎛️ Guidance Scale", callback_data="setting_guidance")],
        [InlineKeyboardButton("🔢 Inference Steps", callback_data="setting_steps")],
        [InlineKeyboardButton("📝 Prompt Enhancer", callback_data="setting_enhancer")],
        [InlineKeyboardButton("⚡ TeaCache", callback_data="setting_teacache")],
        [InlineKeyboardButton("🔙 Asosiy Menyu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    user_id = query.from_user.id
    session = get_user_session(user_id)
    
    data = query.data
    
    if data == "main_menu":
        await query.edit_message_text(
            "🎬 Asosiy Menyu",
            reply_markup=main_menu_keyboard()
        )
    
    elif data == "generate":
        session.is_generating = True
        await query.edit_message_text(
            "🎬 Video yaratishni boshlaymiz!\n\n"
            "Quyidagilardan birini tanlang:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📝 Matn orqali", callback_data="generate_text")],
                [InlineKeyboardButton("🖼️ Rasm orqali", callback_data="generate_image")],
                [InlineKeyboardButton("🔙 Orqaga", callback_data="main_menu")]
            ])
        )
    
    elif data == "settings":
        await query.edit_message_text(
            "⚙️ Sozlamalar - O'zgartirmoqchi bo'lgan parametrni tanlang:",
            reply_markup=settings_keyboard()
        )
    
    elif data == "status":
        await status(update, context)
    
    elif data == "help":
        await help_command(update, context)
    
    elif data == "generate_text":
        await query.edit_message_text(
            "📝 Iltimos, video uchun prompt (tavsif) kiriting:\n\n"
            "Misol: 'A serene lake surrounded by towering mountains, with a few swans gracefully gliding across the water and sunlight dancing on the surface.'"
        )
        context.user_data["waiting_for_prompt"] = True
    
    elif data == "generate_image":
        await query.edit_message_text(
            "🖼️ Iltimos, video yaratish uchun asos bo'ladigan rasmni yuboring."
        )
        context.user_data["waiting_for_image"] = True
    
    elif data == "setting_model_type":
        await query.edit_message_text(
            "🎯 Model turini tanlang:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📝 Matn -> Video", callback_data="set_model_type_text")],
                [InlineKeyboardButton("🖼️ Rasm -> Video", callback_data="set_model_type_image")],
                [InlineKeyboardButton("🔙 Orqaga", callback_data="settings")]
            ])
        )
    
    elif data == "setting_model":
        models = MODEL_ID_CONFIG[session.model_type]
        keyboard = []
        for model in models:
            keyboard.append([InlineKeyboardButton(model.split('/')[-1], callback_data=f"set_model_{model}")])
        keyboard.append([InlineKeyboardButton("🔙 Orqaga", callback_data="settings")])
        
        await query.edit_message_text(
            "🆔 Modelni tanlang:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data == "setting_resolution":
        await query.edit_message_text(
            "📺 Resolutionni tanlang:",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("540P", callback_data="set_res_540P")],
                [InlineKeyboardButton("720P", callback_data="set_res_720P")],
                [InlineKeyboardButton("🔙 Orqaga", callback_data="settings")]
            ])
        )
    
    elif data.startswith("set_"):
        if data.startswith("set_model_type_"):
            session.model_type = "text2video" if "text" in data else "image2video"
            session.model_id = MODEL_ID_CONFIG[session.model_type][0]
            await query.edit_message_text(f"✅ Model turi: {session.model_type} ga o'zgartirildi")
        
        elif data.startswith("set_model_"):
            session.model_id = data.replace("set_model_", "")
            await query.edit_message_text(f"✅ Model: {session.model_id.split('/')[-1]} ga o'zgartirildi")
        
        elif data.startswith("set_res_"):
            session.resolution = data.replace("set_res_", "")
            await query.edit_message_text(f"✅ Resolution: {session.resolution} ga o'zgartirildi")
        
        await asyncio.sleep(1)
        await query.edit_message_text(
            "⚙️ Sozlamalar - O'zgartirmoqchi bo'lgan parametrni tanlang:",
            reply_markup=settings_keyboard()
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = get_user_session(user_id)
    
    if context.user_data.get("waiting_for_prompt"):
        session.prompt = update.message.text
        context.user_data["waiting_for_prompt"] = False
        
        # Video generatsiyani boshlash
        await update.message.reply_text(
            f"✅ Prompt qabul qilindi!\n\n"
            f"📝: {session.prompt}\n\n"
            f"⏳ Video yaratilmoqda... Bu bir necha daqiqa vaqt olishi mumkin."
        )
        
        # Video generatsiya qilish
        await generate_video(update, context, session)
    
    elif context.user_data.get("waiting_for_image"):
        # Rasmni yuklash
        photo_file = await update.message.photo[-1].get_file()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            await photo_file.download_to_drive(tmp_file.name)
            session.image_path = tmp_file.name
        
        context.user_data["waiting_for_image"] = False
        context.user_data["waiting_for_prompt_after_image"] = True
        
        await update.message.reply_text(
            "✅ Rasm qabul qilindi! Endi video uchun prompt (tavsif) kiriting:"
        )

async def generate_video(update: Update, context: ContextTypes.DEFAULT_TYPE, session):
    try:
        user_id = update.effective_user.id
        message = await update.message.reply_text("🔄 Model yuklanmoqda...")
        
        # Prepare arguments
        args = argparse.Namespace()
        args.model_id = session.model_id
        args.resolution = session.resolution
        args.num_frames = session.num_frames
        args.image = session.image_path
        args.guidance_scale = session.guidance_scale
        args.shift = session.shift
        args.inference_steps = session.inference_steps
        args.use_usp = session.use_usp
        args.offload = session.offload
        args.fps = session.fps
        args.seed = session.seed
        args.prompt = session.prompt
        args.prompt_enhancer = session.prompt_enhancer
        args.teacache = session.teacache
        args.teacache_thresh = session.teacache_thresh
        args.use_ret_steps = session.use_ret_steps
        args.outdir = "telegram_videos"
        
        # Download model
        await message.edit_text("📥 Model yuklanmoqda...")
        args.model_id = download_model(args.model_id)
        
        # Set resolution
        if args.resolution == "540P":
            height = 544
            width = 960
        elif args.resolution == "720P":
            height = 720
            width = 1280
        
        # Load image if provided
        image = load_image(args.image).convert("RGB") if args.image else None
        negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        
        # Initialize pipeline
        await message.edit_text("🔧 Pipeline tayyorlanmoqda...")
        if image is None:
            pipe = Text2VideoPipeline(
                model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
            )
        else:
            pipe = Image2VideoPipeline(
                model_path=args.model_id, dit_path=args.model_id, use_usp=args.use_usp, offload=args.offload
            )
            args.image = load_image(args.image)
            image_width, image_height = args.image.size
            if image_height > image_width:
                height, width = width, height
            args.image = resizecrop(args.image, height, width)
        
        # Initialize TeaCache if requested
        if args.teacache:
            pipe.transformer.initialize_teacache(enable_teacache=True, num_steps=args.inference_steps, 
                                                 teacache_thresh=args.teacache_thresh, use_ret_steps=args.use_ret_steps, 
                                                 ckpt_dir=args.model_id)
        
        # Prepare generation kwargs
        kwargs = {
            "prompt": args.prompt,
            "negative_prompt": negative_prompt,
            "num_frames": args.num_frames,
            "num_inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "shift": args.shift,
            "generator": torch.Generator(device="cuda").manual_seed(args.seed),
            "height": height,
            "width": width,
        }
        
        if image is not None:
            kwargs["image"] = args.image.convert("RGB")
        
        # Generate video
        await message.edit_text("🎬 Video yaratilmoqda... Bu biroz vaqt olishi mumkin.")
        with torch.cuda.amp.autocast(dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(**kwargs)[0]
        
        # Save video
        await message.edit_text("💾 Video saqlanmoqda...")
        os.makedirs("telegram_videos", exist_ok=True)
        output_path = f"telegram_videos/video_{user_id}_{int(time.time())}.mp4"
        
        imageio.mimwrite(
            output_path, 
            video_frames, 
            fps=args.fps,
            codec='libx264',
            ffmpeg_params=['-crf', '23', '-preset', 'medium', '-pix_fmt', 'yuv420p']
        )
        
        # Send video to user
        await message.edit_text("📤 Video yuborilmoqda...")
        with open(output_path, 'rb') as video_file:
            await update.message.reply_video(
                video=video_file,
                caption=f"🎬 SkyReels tomonidan yaratildi\n\n📝: {args.prompt}\n⚙️: {args.resolution}, {args.num_frames} frames, {args.fps} FPS",
                reply_markup=main_menu_keyboard()
            )
        
        # Cleanup
        await message.delete()
        if os.path.exists(output_path):
            os.remove(output_path)
        if session.image_path and os.path.exists(session.image_path):
            os.remove(session.image_path)
        
        session.is_generating = False
        
    except Exception as e:
        await update.message.reply_text(
            f"❌ Xatolik yuz berdi: {str(e)}\n\n"
            f"Iltimos, qaytadan urinib ko'ring yoki sozlamalarni tekshiring.",
            reply_markup=main_menu_keyboard()
        )
        session.is_generating = False
        logger.error(f"Error generating video: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

def main():
    # Botni yaratish
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("generate", start))
    application.add_handler(CommandHandler("settings", start))
    
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_message))
    
    application.add_error_handler(error_handler)
    
    # Botni ishga tushirish
    print("Bot ishga tushdi...")
    application.run_polling()

if __name__ == "__main__":
    main()
