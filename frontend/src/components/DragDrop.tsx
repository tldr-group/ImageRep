import React, { useContext, useEffect, useRef, useState } from "react";
import { DragDropProps } from "./interfaces";

export const dragDropStyle = {
    height: '75vh', width: '75vw',
    outline: '10px dashed #b5bab6', color: '#b5bab6',
    fontSize: '3.5em', justifyContent: 'center', alignItems: 'center',
    borderRadius: '10px', padding: '10px', display: 'flex', margin: 'auto'
}
const flexCenterClasses = "flex items-center justify-center";

const DragDrop = ({ loadFromFile }: DragDropProps): JSX.Element => {
    // Drag and drop for file upload
    const handleDrag = (e: any) => { e.preventDefault(); }
    const handeDrop = (e: any) => {
        e.preventDefault();
        if (e.dataTransfer.items) {
            const item = e.dataTransfer.items[0];
            if (item.kind === "file") {
                const file = item.getAsFile();
                loadFromFile(file);
            };
        };
    };
    return (
        <div style={dragDropStyle}
            onDragOver={handleDrag}
            onDrop={handeDrop}
        >
            <span>Drag microstructure file!</span>
        </div>
    );
}

export default DragDrop