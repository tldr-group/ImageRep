import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { MenuState } from "./interfaces";

import Toast from 'react-bootstrap/Toast'
import ToastContainer from "react-bootstrap/ToastContainer";
import Button from 'react-bootstrap/Button';


const PhaseSelect = () => {
    return (
        <>
            <span>Choose phase to analyze representativity of:</span>
            <Button variant="dark">Confirm</Button>
        </>
    );
}


const getMenuInfo = (state: MenuState) => {
    switch (state) {
        case 'phase':
            return { title: "Select Phase!", innerHTML: <PhaseSelect /> }
        case 'hidden': // fall through
        default:
            return { title: "", innerHTML: <></> }
    }
}

const Menu = () => {
    const {
        menuState: [menuState,]
    } = useContext(AppContext)!

    return (
        <>
            <ToastContainer className="p-5" position="bottom-end">
                <Toast show={menuState != 'hidden'}>
                    <Toast.Header className="roundedme-2">
                        <strong className="me-auto" style={{ fontSize: '1.5em' }}>{getMenuInfo(menuState).title}</strong>
                    </Toast.Header>
                    <Toast.Body>
                        {getMenuInfo(menuState).innerHTML}
                    </Toast.Body>
                </Toast>
            </ToastContainer>
        </>
    )
}

export default Menu;