// page.tsx
"use client";

import React, { useState, useRef } from "react";
import Head from "next/head";
import styles from "./page.module.css";

interface ImageInfo {
  src: string;
  file: File | null;
}

const Page = () => {
  const [imageInfo, setImageInfo] = useState<ImageInfo>({
    src: "",
    file: null,
  });
  const [categoryId, setCategoryId] = useState(1); // 新增分类id的状态
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

  // 修改或添加到您的现有 page.tsx 中
  const addCategory = async () => {
    if (imageInfo.file) {
      const formData = new FormData();
      formData.append('image', imageInfo.file); // 添加文件对象
      formData.append('id', String(categoryId)); // 添加分类ID
  
      try {
        const response = await fetch('http://localhost:4000/uploadImage', {
          method: 'POST',
          body: formData, // 直接发送 FormData，不需要设置 'Content-Type'，浏览器会自动处理
          // 注意：这里不设置 headers，让浏览器自动设置 Content-Type 为 multipart/form-data 并正确包含 boundary
        });
  
        if (!response.ok) {
          throw new Error(`Server responded with ${response.status}`);
        }
  
        const data = await response.json();
        console.log(data); // 处理响应
  
        // 更新分类id
        setCategoryId(prev => prev + 1);
      } catch (error) {
        console.error("Upload failed", error);
      }
    }
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
        style={{
          backgroundImage: imageInfo.src ? `url(${imageInfo.src})` : "",
          backgroundSize: "cover",
        }}
      >
        {imageInfo.src ? "" : "拖拽图片到这里或点击进行拍照"}
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef} // 将ref赋给input元素
          style={{ display: "none" }}
          onChange={handleFiles}
        />
      </div>
      <div style={{ textAlign: "center" }}>
        <button className={styles.button}>识别</button>
        <button className={styles.button} onClick={addCategory}>
          增加分类
        </button>
      </div>
    </>
  );
};

export default Page;
