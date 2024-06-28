import React, { useContext, useEffect, useRef, useState } from "react";
import AppContext, { IR_LIMIT_PX, MenuState } from "./interfaces";

import Modal from 'react-bootstrap/Modal';
import Accordion from 'react-bootstrap/Accordion';
import Toast from 'react-bootstrap/Toast'
import ToastContainer from "react-bootstrap/ToastContainer";
import Button from 'react-bootstrap/Button';



export const ErrorMessage = () => {
    const {
        errorState: [errorObject, setErrorObject]
    } = useContext(AppContext)!;


    const handleClose = () => { setErrorObject({ msg: "", stackTrace: "" }) };

    return (
        <>
            <Modal show={errorObject.msg !== ""} onHide={handleClose}>
                <Modal.Header style={{ backgroundColor: '#eb4034', color: '#ffffff' }} closeVariant="white" closeButton>
                    <Modal.Title>Error</Modal.Title>
                </Modal.Header>
                <Modal.Body>{errorObject.msg}</Modal.Body>
                <Modal.Body>
                    <Accordion defaultActiveKey="0">
                        <Accordion.Item eventKey="1" key={1}>
                            <Accordion.Header>Stack trace</Accordion.Header>
                            {/*Need to manually overwrite the style here because of werid bug*/}
                            <Accordion.Body style={{ visibility: "visible" }}>
                                {errorObject.stackTrace}
                            </Accordion.Body>
                        </Accordion.Item>
                    </Accordion>
                </Modal.Body>
                <Modal.Footer>
                    <Button variant="dark" onClick={handleClose}>
                        Understood!
                    </Button>
                </Modal.Footer>
            </Modal >
        </>
    );
}


export const CLSModal = () => {
    const {
        analysisInfo: [analysisInfo,],
        showWarning: [, setShowWarning],
    } = useContext(AppContext)!;

    const hide = () => {
        setShowWarning(false);
    }

    return (
        <>
            <ToastContainer className="p-5" position="bottom-start">
                <Toast onClose={(e) => hide()}>
                    <Toast.Header className="roundedme-2" closeButton={true} style={{ backgroundColor: '#fcba03', color: '#ffffff' }}>
                        <strong className="me-auto" style={{ fontSize: '1.5em' }}>Warning!</strong>
                    </Toast.Header>
                    <Toast.Body>
                        Integral Range/feature size of {analysisInfo?.integralRange.toFixed(2)} exceeds tested limit of {IR_LIMIT_PX}px, results may be inaccurate.
                    </Toast.Body>
                </Toast>
            </ToastContainer>
        </>
    );
}