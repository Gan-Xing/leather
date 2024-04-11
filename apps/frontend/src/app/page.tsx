// page.tsx
"use client"

import React, { useState, useRef } from 'react';
import Head from 'next/head';
import styles from './page.module.css';

interface ImageInfo {
  src: string;
  file: File | null;
}

const Page = () => {
  const [imageInfo, setImageInfo] = useState<ImageInfo>({ src: '', file: null });
  const fileInputRef = useRef<HTMLInputElement>(null); // 使用ref创建引用

  const handleFiles = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      const imgSrc = URL.createObjectURL(file);
      setImageInfo({ src: imgSrc, file });
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    handleFiles({ target: { files } } as React.ChangeEvent<HTMLInputElement>);
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click(); // 触发文件输入元素的点击事件
  };

  return (
    <>
      <Head>
        <title>图片上传和拍照页面</title>
      </Head>
      <div 
        className={styles.uploadArea}
        onClick={triggerFileInput} // 添加点击事件处理器
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        style={{ backgroundImage: imageInfo.src ? `url(${imageInfo.src})` : '' }}
      >
        {imageInfo.src ? '' : '拖拽图片到这里或点击进行拍照'}
        <input 
          type="file" 
          accept="image/*" 
          ref={fileInputRef} // 将ref赋给input元素
          style={{ display: 'none' }} 
          onChange={handleFiles}
        />
      </div>
      <div style={{ textAlign: 'center' }}>
        <button className={styles.button}>识别</button>
        <button className={styles.button}>增加分类</button>
      </div>
    </>
  );
};

export default Page;
